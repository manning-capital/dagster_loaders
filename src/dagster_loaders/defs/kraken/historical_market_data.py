import os
import re
import json
import pathlib
import zipfile
import datetime as dt
import tempfile
from typing import Final, Literal
from collections.abc import Iterator

import pandas as pd
from dagster import (
    Backoff,
    Failure,
    Definitions,
    RetryPolicy,
    MetadataValue,
    BackfillPolicy,
    AssetMaterialization,
    AssetExecutionContext,
    DailyPartitionsDefinition,
    asset,
)
from google.oauth2 import service_account
from googleapiclient.http import MediaIoBaseDownload
from mc_postgres_db.models import ProviderAssetMarket
from googleapiclient.discovery import build
from mc_postgres_db.operations import set_data

from dagster_loaders.resources import PostgresResource
from dagster_loaders.defs.provider_assets import provider_asset_map
from dagster_loaders.defs.kraken.market_data import ASSET_PAIRS_URL, _request_kraken

KRAKEN_DRIVE_FOLDER_ID: Final[str] = "15RSlNuW_h0kVM8or8McOGOMfHeBFvFGI"
KRAKEN_FULL_HISTORY_DRIVE_FILE_ID: Final[str] = "1ptNqWYidLkhb2VAKuLCxmp2OXEfGO-AP"
DAILY_START: Final[dt.date] = dt.date(2014, 1, 1)
BATCH_SIZE: int = 5000
DOWNLOAD_CHUNK_BYTES: int = 16 * 1024 * 1024
SA_JSON_ENV_VAR: Final[str] = "KRAKEN_DRIVE_SA_JSON"

_ONE_MIN_CSV_RE: Final[re.Pattern[str]] = re.compile(r"^([A-Z0-9]+)_1\.csv$")
_QUARTERLY_ZIP_RE: Final[re.Pattern[str]] = re.compile(
    r"^Kraken_OHLCVT_Q([1-4])_(\d{4})\.zip$"
)

ZipKey = tuple[Literal["full_history", "quarterly"], str]

daily_partitions = DailyPartitionsDefinition(start_date=DAILY_START.isoformat())


def _quarter_to_filename(quarter: str) -> str:
    year, q = quarter.split("-")
    return f"Kraken_OHLCVT_{q}_{year}.zip"


def _quarter_for_date(date: dt.date) -> tuple[int, int]:
    return date.year, (date.month - 1) // 3 + 1


def _drive_service(sa_info: dict):
    creds = service_account.Credentials.from_service_account_info(
        sa_info, scopes=["https://www.googleapis.com/auth/drive.readonly"]
    )
    return build("drive", "v3", credentials=creds, cache_discovery=False)


def _resolve_drive_file_id(service, folder_id: str, filename: str) -> str:
    resp = (
        service.files()
        .list(
            q=f"'{folder_id}' in parents and name = '{filename}' and trashed = false",
            fields="files(id, name, size)",
            pageSize=10,
            supportsAllDrives=True,
            includeItemsFromAllDrives=True,
        )
        .execute()
    )
    files = resp.get("files", [])
    if not files:
        raise FileNotFoundError(f"{filename} not found in Drive folder {folder_id}")
    return files[0]["id"]


def _list_available_quarters(service, folder_id: str) -> set[tuple[int, int]]:
    available: set[tuple[int, int]] = set()
    page_token: str | None = None
    while True:
        resp = (
            service.files()
            .list(
                q=f"'{folder_id}' in parents and trashed = false",
                fields="nextPageToken, files(id, name)",
                pageSize=200,
                pageToken=page_token,
                supportsAllDrives=True,
                includeItemsFromAllDrives=True,
            )
            .execute()
        )
        for f in resp.get("files", []):
            m = _QUARTERLY_ZIP_RE.match(f["name"])
            if m:
                q, year = int(m.group(1)), int(m.group(2))
                available.add((year, q))
        page_token = resp.get("nextPageToken")
        if not page_token:
            break
    return available


def _zip_key_for_date(
    date: dt.date, available_quarters: set[tuple[int, int]]
) -> ZipKey:
    yq = _quarter_for_date(date)
    if yq in available_quarters:
        return ("quarterly", _quarter_to_filename(f"{yq[0]}-Q{yq[1]}"))
    return ("full_history", KRAKEN_FULL_HISTORY_DRIVE_FILE_ID)


def _group_dates_by_zip(
    dates: set[dt.date], available_quarters: set[tuple[int, int]]
) -> dict[ZipKey, set[dt.date]]:
    out: dict[ZipKey, set[dt.date]] = {}
    for d in dates:
        key = _zip_key_for_date(d, available_quarters)
        out.setdefault(key, set()).add(d)
    return out


def _download_zip(service, file_id: str, dest: pathlib.Path) -> None:
    request = service.files().get_media(fileId=file_id, supportsAllDrives=True)
    with dest.open("wb") as fh:
        downloader = MediaIoBaseDownload(fh, request, chunksize=DOWNLOAD_CHUNK_BYTES)
        done = False
        while not done:
            _status, done = downloader.next_chunk()


def _resolve_zip_path(
    service, zip_key: ZipKey, dest_dir: pathlib.Path
) -> tuple[pathlib.Path, str]:
    """Download the zip identified by `zip_key` into `dest_dir` and return
    `(path, label)`. `label` is a short descriptor for logging/metadata."""
    kind, ref = zip_key
    if kind == "quarterly":
        filename = ref
        file_id = _resolve_drive_file_id(service, KRAKEN_DRIVE_FOLDER_ID, filename)
        label = filename
    else:
        file_id = ref
        filename = "kraken_full_history.zip"
        label = "full_history"
    dest = dest_dir / filename
    _download_zip(service, file_id, dest)
    return dest, label


def _list_one_minute_altnames(zip_path: pathlib.Path) -> list[str]:
    with zipfile.ZipFile(zip_path) as zf:
        result: list[str] = []
        for name in zf.namelist():
            m = _ONE_MIN_CSV_RE.match(pathlib.PurePosixPath(name).name)
            if m:
                result.append(m.group(1))
        return result


def _iter_one_minute_csvs(
    zip_path: pathlib.Path,
) -> Iterator[tuple[str, pd.DataFrame]]:
    with zipfile.ZipFile(zip_path) as zf:
        for name in zf.namelist():
            m = _ONE_MIN_CSV_RE.match(pathlib.PurePosixPath(name).name)
            if not m:
                continue
            altname = m.group(1)
            with zf.open(name) as fh:
                df = pd.read_csv(
                    fh,
                    header=None,
                    names=[
                        "timestamp",
                        "open",
                        "high",
                        "low",
                        "close",
                        "volume",
                        "trades",
                    ],
                )
            yield altname, df


def _resolve_partition_keys(context: AssetExecutionContext) -> list[str]:
    try:
        return list(context.partition_keys)
    except Exception:
        pass
    try:
        return list(
            daily_partitions.get_partition_keys_in_range(context.partition_key_range)
        )
    except Exception:
        pass
    return [context.partition_key]


def _materialize_historical(
    context: AssetExecutionContext,
    postgres: PostgresResource,
    dates: set[dt.date],
) -> Iterator[tuple[dt.date, dict]]:
    """Core implementation. Yields `(date, metadata_dict)` for each
    successfully materialized partition as it is finished, then raises
    `dagster.Failure` if any requested partition produced zero rows. Exposed
    separately so tests can drive multi-date runs without needing Dagster's
    full single-run-backfill machinery."""
    as_of = dt.date.today()
    sa_info = json.loads(os.environ[SA_JSON_ENV_VAR])

    engine = postgres.get_engine()
    empty_dates: list[dt.date] = []
    materialized_count = 0
    total_rows = 0
    try:
        provider_id, asset_map = provider_asset_map(engine, "Kraken", as_of)
        context.log.info(f"[fetch] provider_assets count={len(asset_map)}")

        pairs_body = _request_kraken(ASSET_PAIRS_URL)
        altname_map: dict[str, tuple[str, str]] = {
            info["altname"]: (info["base"], info["quote"])
            for info in pairs_body["result"].values()
            if info.get("execution_venue", "international") == "international"
            and "altname" in info
        }
        context.log.info(f"[fetch] altname_map count={len(altname_map)}")

        service = _drive_service(sa_info)
        available_quarters = _list_available_quarters(service, KRAKEN_DRIVE_FOLDER_ID)
        context.log.info(f"[fetch] available_quarters={sorted(available_quarters)}")

        groups = _group_dates_by_zip(dates, available_quarters)
        context.log.info(
            f"[fetch] zip_groups={ {k[1]: len(v) for k, v in groups.items()} }"
        )

        for zip_key, dates_for_this_zip in groups.items():
            with tempfile.TemporaryDirectory() as tmp:
                tmp_path = pathlib.Path(tmp)
                if zip_key[0] == "full_history":
                    context.log.warning(
                        f"[fetch] downloading the full-history zip — this is "
                        f"~7.3 GB compressed and covers every date whose "
                        f"quarter is not yet in the per-quarter Drive folder "
                        f"({len(dates_for_this_zip)} requested). Expect "
                        f"several minutes for the download plus additional "
                        f"time to decode each per-pair CSV; large pairs (e.g. "
                        f"XBTUSD spanning ~9 years) decompress to hundreds of "
                        f"MB each."
                    )
                zip_path, zip_label = _resolve_zip_path(service, zip_key, tmp_path)
                size_mb = zip_path.stat().st_size / 1e6
                context.log.info(
                    f"[fetch] zip={zip_label} size_mb={size_mb:.1f} "
                    f"dates={len(dates_for_this_zip)} "
                    f"(starting per-pair decode)"
                )

                per_partition_frames: dict[dt.date, list[pd.DataFrame]] = {
                    d: [] for d in dates_for_this_zip
                }

                kept: set[str] = set()
                skip_no_altname: list[str] = []
                skip_no_asset: list[str] = []
                for altname in _list_one_minute_altnames(zip_path):
                    pair_codes = altname_map.get(altname)
                    if pair_codes is None:
                        skip_no_altname.append(altname)
                        continue
                    base_code, quote_code = pair_codes
                    if base_code not in asset_map or quote_code not in asset_map:
                        skip_no_asset.append(f"{altname}({base_code}/{quote_code})")
                        continue
                    kept.add(altname)
                if skip_no_altname or skip_no_asset:
                    context.log.info(
                        f"[skip] zip={zip_label} "
                        f"not_in_altname_map={sorted(skip_no_altname)} "
                        f"missing_asset={sorted(skip_no_asset)}"
                    )

                for altname, raw_df in _iter_one_minute_csvs(zip_path):
                    if altname not in kept:
                        continue
                    pair_codes = altname_map[altname]
                    base_code, quote_code = pair_codes

                    n_total = len(raw_df)
                    df = raw_df.drop(columns=["trades"])
                    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
                    row_dates = df["timestamp"].dt.date
                    df = df[row_dates.isin(dates_for_this_zip)].copy()
                    if df.empty:
                        context.log.info(f"[parse] altname={altname} rows=0/{n_total}")
                        continue

                    for c in ("open", "high", "low", "close", "volume"):
                        df[c] = df[c].astype(float)
                    df["from_asset_id"] = asset_map[quote_code]
                    df["to_asset_id"] = asset_map[base_code]
                    df["provider_id"] = provider_id

                    df_dates = df["timestamp"].dt.date
                    for date, sub in df.groupby(df_dates):
                        per_partition_frames[date].append(sub)
                    context.log.info(
                        f"[parse] altname={altname} rows={len(df)}/{n_total}"
                    )

                for date in sorted(dates_for_this_zip):
                    frames = per_partition_frames[date]
                    if not frames:
                        empty_dates.append(date)
                        continue
                    combined = pd.concat(frames, ignore_index=True)
                    combined = combined.drop_duplicates(
                        subset=[
                            "timestamp",
                            "provider_id",
                            "from_asset_id",
                            "to_asset_id",
                        ],
                        keep="last",
                    )
                    n = len(combined)
                    n_batches = 0
                    for i in range(0, n, BATCH_SIZE):
                        batch = combined.iloc[i : i + BATCH_SIZE]
                        set_data(
                            engine, ProviderAssetMarket.__tablename__, batch, "upsert"
                        )
                        n_batches += 1
                    pair_count = len(frames)
                    context.log.info(
                        f"[upsert] partition={date.isoformat()} rows={n} "
                        f"batches={n_batches} pairs={pair_count} zip={zip_label}"
                    )
                    metadata = {
                        "row_count": n,
                        "pair_count": pair_count,
                        "min_timestamp": MetadataValue.text(
                            combined["timestamp"].min().isoformat()
                        ),
                        "max_timestamp": MetadataValue.text(
                            combined["timestamp"].max().isoformat()
                        ),
                        "source_zip": MetadataValue.text(zip_label),
                        "table": ProviderAssetMarket.__tablename__,
                    }
                    yield date, metadata
                    materialized_count += 1
                    total_rows += n

        context.log.info(
            f"[done] partitions_materialized={materialized_count} "
            f"empty={len(empty_dates)} total_rows={total_rows}"
        )

        if empty_dates:
            missing = [d.isoformat() for d in sorted(empty_dates)]
            raise Failure(description=f"No rows found for partition(s): {missing}")
    finally:
        engine.dispose()


@asset(
    partitions_def=daily_partitions,
    backfill_policy=BackfillPolicy.single_run(),
    pool="kraken-historical",
    group_name="market_data",
    kinds={"python", "postgres"},
    owners=["glynfinck@gmail.com"],
    tags={
        "domain": "market-data",
        "provider": "kraken",
        "source": "historical-csv",
    },
    retry_policy=RetryPolicy(max_retries=3, delay=5.0, backoff=Backoff.EXPONENTIAL),
    description=(
        "Kraken historical 1-min OHLCVT loaded from Google Drive zips into "
        "provider_asset_market. One partition per calendar day. Each run "
        "groups its requested dates by source zip — preferring the per-quarter "
        "zip when present in the Drive folder, and falling back to the "
        "all-history zip for any date whose quarter has not been published — "
        "so each unique zip is downloaded at most once per run. Per-partition "
        "MaterializeResults are yielded as soon as each zip finishes, so the "
        "Dagster UI updates partition status incrementally. Pair-code mapping "
        "(altname -> base/quote) is resolved at run time from the public "
        "AssetPairs endpoint and joined against the active provider_asset rows "
        "for the Kraken provider. Rows are deduped on the same PK as the API "
        "loader and upserted in batches of 5000. A run fails (after yielding "
        "successful partitions) if any requested partition produced zero rows."
    ),
)
def kraken_provider_asset_market_historical(
    context: AssetExecutionContext, postgres: PostgresResource
) -> None:
    partition_keys = _resolve_partition_keys(context)
    dates = {dt.date.fromisoformat(k) for k in partition_keys}
    for date, metadata in _materialize_historical(context, postgres, dates):
        context.log_event(
            AssetMaterialization(
                asset_key=context.asset_key,
                partition=date.isoformat(),
                metadata=metadata,
            )
        )


defs = Definitions(assets=[kraken_provider_asset_market_historical])
