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
    AssetKey,
    Definitions,
    RetryPolicy,
    MetadataValue,
    BackfillPolicy,
    AssetCheckResult,
    AssetRecordsFilter,
    AssetMaterialization,
    AssetExecutionContext,
    DailyPartitionsDefinition,
    AssetCheckExecutionContext,
    asset,
    asset_check,
)
from sqlalchemy import func, select
from google.oauth2 import service_account
from sqlalchemy.orm import Session
from googleapiclient.http import MediaIoBaseDownload
from mc_postgres_db.models import Provider, ProviderAssetMarket
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


def _count_partition_rows(session: Session, provider_id: int, date: dt.date) -> int:
    """Count rows in `provider_asset_market` for a Kraken partition on a UTC
    calendar day. Used by both the asset (post-upsert sanity log) and by the
    asset checks."""
    start = dt.datetime.combine(date, dt.time.min, tzinfo=dt.timezone.utc)
    end = start + dt.timedelta(days=1)
    return int(
        session.execute(
            select(func.count())
            .select_from(ProviderAssetMarket)
            .where(ProviderAssetMarket.provider_id == provider_id)
            .where(ProviderAssetMarket.timestamp >= start)
            .where(ProviderAssetMarket.timestamp < end)
        ).scalar_one()
    )


def _process_zip_for_dates(
    context: AssetExecutionContext,
    engine,
    zip_path: pathlib.Path,
    zip_label: str,
    dates_for_this_zip: set[dt.date],
    altname_map: dict[str, tuple[str, str]],
    asset_map: dict[str, int],
    provider_id: int,
) -> Iterator[tuple[dt.date | None, dict | list[dt.date]]]:
    """Process a single downloaded zip for a set of dates. Yields
    ``(date, metadata)`` tuples for each non-empty date in sorted order, then
    finally yields ``(None, empty_dates)`` listing dates that produced zero
    rows from this zip. Caller emits the materialization events and decides
    what to do with empty dates."""
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

    per_partition_frames: dict[dt.date, list[pd.DataFrame]] = {
        d: [] for d in dates_for_this_zip
    }

    for altname, raw_df in _iter_one_minute_csvs(zip_path):
        if altname not in kept:
            continue
        base_code, quote_code = altname_map[altname]

        n_total = len(raw_df)
        df = raw_df.drop(columns=["trades"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
        row_dates = df["timestamp"].dt.date
        df = df[row_dates.isin(dates_for_this_zip)].copy()
        if df.empty:
            continue

        for c in ("open", "high", "low", "close", "volume"):
            df[c] = df[c].astype(float)
        df["from_asset_id"] = asset_map[quote_code]
        df["to_asset_id"] = asset_map[base_code]
        df["provider_id"] = provider_id

        df_dates = df["timestamp"].dt.date
        for date, sub in df.groupby(df_dates):
            per_partition_frames[date].append(sub)
        context.log.info(f"[parse] altname={altname} rows={len(df)}/{n_total}")

    empty: list[dt.date] = []
    for date in sorted(dates_for_this_zip):
        frames = per_partition_frames[date]
        if not frames:
            empty.append(date)
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
        non_zero_volume_count = int((combined["volume"] > 0).sum())
        n_batches = 0
        for i in range(0, n, BATCH_SIZE):
            batch = combined.iloc[i : i + BATCH_SIZE]
            set_data(engine, ProviderAssetMarket.__tablename__, batch, "upsert")
            n_batches += 1
        pair_count = len(frames)
        context.log.info(
            f"[upsert] partition={date.isoformat()} rows={n} "
            f"non_zero_volume_rows={non_zero_volume_count} "
            f"batches={n_batches} pairs={pair_count} zip={zip_label}"
        )
        metadata = {
            "row_count": n,
            "non_zero_volume_count": non_zero_volume_count,
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

    yield None, empty


def _download_and_process_zip(
    context: AssetExecutionContext,
    engine,
    service,
    zip_key: ZipKey,
    dates_for_this_zip: set[dt.date],
    altname_map: dict[str, tuple[str, str]],
    asset_map: dict[str, int],
    provider_id: int,
) -> Iterator[tuple[dt.date | None, dict | list[dt.date]]]:
    """Download `zip_key` and process the requested dates against it. Yields
    `(date, metadata)` for non-empty dates, then `(None, empty_dates)`."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = pathlib.Path(tmp)
        if zip_key[0] == "full_history":
            context.log.warning(
                f"[fetch] downloading the full-history zip — this is "
                f"~7.3 GB compressed and covers every date whose quarter is "
                f"not yet in the per-quarter Drive folder "
                f"({len(dates_for_this_zip)} requested). Expect several "
                f"minutes for the download plus additional time to decode "
                f"each per-pair CSV; large pairs (e.g. XBTUSD spanning "
                f"~9 years) decompress to hundreds of MB each."
            )
        zip_path, zip_label = _resolve_zip_path(service, zip_key, tmp_path)
        size_mb = zip_path.stat().st_size / 1e6
        context.log.info(
            f"[fetch] zip={zip_label} size_mb={size_mb:.1f} "
            f"dates={len(dates_for_this_zip)} (starting per-pair decode)"
        )
        yield from _process_zip_for_dates(
            context,
            engine,
            zip_path,
            zip_label,
            dates_for_this_zip,
            altname_map,
            asset_map,
            provider_id,
        )


def _materialize_historical(
    context: AssetExecutionContext,
    postgres: PostgresResource,
    dates: set[dt.date],
) -> Iterator[tuple[dt.date, dict]]:
    """Core implementation. Yields `(date, metadata_dict)` for each
    successfully materialized partition as it is finished, then raises
    `dagster.Failure` if any requested partition produced zero rows even
    after falling back from a quarterly zip to the full-history zip. Exposed
    separately so tests can drive multi-date runs without needing Dagster's
    full single-run-backfill machinery."""
    as_of = dt.date.today()
    sa_info = json.loads(os.environ[SA_JSON_ENV_VAR])

    engine = postgres.get_engine()
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
        full_history_key: ZipKey = (
            "full_history",
            KRAKEN_FULL_HISTORY_DRIVE_FILE_ID,
        )
        full_history_dates: set[dt.date] = set(groups.pop(full_history_key, set()))

        plan_summary = {k[1]: len(v) for k, v in groups.items()}
        if full_history_dates:
            plan_summary["full_history"] = len(full_history_dates)
        context.log.info(
            f"[fetch] plan: zips={list(plan_summary.keys())} "
            f"counts={plan_summary} (quarterly zips first; any quarterly "
            f"dates with no rows will fall back to the full-history zip)"
        )

        # Process quarterly zips first; any empty dates become candidates
        # for the full-history fallback.
        fallback_dates: set[dt.date] = set()
        for zip_key, dates_for_this_zip in groups.items():
            for date_or_none, payload in _download_and_process_zip(
                context,
                engine,
                service,
                zip_key,
                dates_for_this_zip,
                altname_map,
                asset_map,
                provider_id,
            ):
                if date_or_none is None:
                    empties = payload  # type: ignore[assignment]
                    if empties:
                        context.log.info(
                            f"[fallback] zip={zip_key[1]} empty_dates="
                            f"{[d.isoformat() for d in empties]} -> queued "
                            f"for full-history fallback"
                        )
                        fallback_dates.update(empties)
                else:
                    yield date_or_none, payload  # type: ignore[misc]
                    materialized_count += 1
                    total_rows += payload["row_count"]  # type: ignore[index]

        # Single full-history pass: union of originally-routed dates and
        # quarterly fallbacks. Downloads at most once.
        all_full_history_dates = full_history_dates | fallback_dates
        truly_empty: list[dt.date] = []
        if all_full_history_dates:
            for date_or_none, payload in _download_and_process_zip(
                context,
                engine,
                service,
                full_history_key,
                all_full_history_dates,
                altname_map,
                asset_map,
                provider_id,
            ):
                if date_or_none is None:
                    truly_empty.extend(payload)  # type: ignore[arg-type]
                else:
                    yield date_or_none, payload  # type: ignore[misc]
                    materialized_count += 1
                    total_rows += payload["row_count"]  # type: ignore[index]

        context.log.info(
            f"[done] partitions_materialized={materialized_count} "
            f"empty={len(truly_empty)} total_rows={total_rows}"
        )

        if truly_empty:
            missing = [d.isoformat() for d in sorted(truly_empty)]
            raise Failure(description=f"No rows found for partition(s): {missing}")
    finally:
        engine.dispose()


PARTITION_HAS_NON_ZERO_VOLUME_CHECK: Final[str] = "partition_has_non_zero_volume"
PARTITION_DB_COUNT_AT_LEAST_MATERIALIZED_CHECK: Final[str] = (
    "partition_db_count_at_least_materialized"
)


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
        # Stream a per-partition AssetMaterialization event so the run log
        # shows each partition completing in real time as the asset processes
        # them. Dagster also auto-emits an empty synthetic materialization
        # at end-of-step for the asset's `Nothing` output; the asset check
        # tolerates that by scanning recent materializations and picking the
        # one that actually carries `row_count`.
        context.log_event(
            AssetMaterialization(
                asset_key=context.asset_key,
                partition=date.isoformat(),
                metadata=metadata,
            )
        )


def _resolve_check_partition_keys(context: AssetCheckExecutionContext) -> list[str]:
    """Return the list of partition keys covered by this check invocation,
    handling both single-partition runs and multi-partition single-run
    backfills (where `context.partition_key` is unavailable)."""
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


def _count_non_zero_volume_rows(
    session: Session, provider_id: int, date: dt.date
) -> tuple[int, int]:
    start = dt.datetime.combine(date, dt.time.min, tzinfo=dt.timezone.utc)
    end = start + dt.timedelta(days=1)
    non_zero = session.execute(
        select(func.count())
        .select_from(ProviderAssetMarket)
        .where(ProviderAssetMarket.provider_id == provider_id)
        .where(ProviderAssetMarket.timestamp >= start)
        .where(ProviderAssetMarket.timestamp < end)
        .where(ProviderAssetMarket.volume > 0)
    ).scalar_one()
    total = session.execute(
        select(func.count())
        .select_from(ProviderAssetMarket)
        .where(ProviderAssetMarket.provider_id == provider_id)
        .where(ProviderAssetMarket.timestamp >= start)
        .where(ProviderAssetMarket.timestamp < end)
    ).scalar_one()
    return int(non_zero), int(total)


@asset_check(
    asset=kraken_provider_asset_market_historical,
    name=PARTITION_HAS_NON_ZERO_VOLUME_CHECK,
    description=(
        "For each materialized partition, verify the partition has at least "
        "one row with volume > 0 in provider_asset_market for the Kraken "
        "provider. Catches partitions that landed only zero-volume ticks."
    ),
)
def check_partition_has_non_zero_volume(
    context: AssetCheckExecutionContext, postgres: PostgresResource
) -> AssetCheckResult:
    partition_keys = _resolve_check_partition_keys(context)

    engine = postgres.get_engine()
    try:
        with Session(engine) as session:
            provider_id = session.execute(
                select(Provider.id).where(Provider.name == "Kraken")
            ).scalar_one()

            per_partition_counts: dict[str, dict[str, int]] = {}
            failed: list[str] = []
            total_non_zero = 0
            total_rows = 0
            for pk in partition_keys:
                non_zero, total = _count_non_zero_volume_rows(
                    session, provider_id, dt.date.fromisoformat(pk)
                )
                per_partition_counts[pk] = {
                    "non_zero_volume_count": non_zero,
                    "row_count": total,
                }
                total_non_zero += non_zero
                total_rows += total
                if non_zero == 0:
                    failed.append(pk)
                context.log.info(
                    f"[check] partition={pk} non_zero_volume_count={non_zero} "
                    f"row_count={total} passed={non_zero > 0}"
                )

        if len(partition_keys) == 1:
            pk = partition_keys[0]
            counts = per_partition_counts[pk]
            return AssetCheckResult(
                passed=pk not in failed,
                metadata={
                    "partition": pk,
                    "non_zero_volume_count": counts["non_zero_volume_count"],
                    "row_count": counts["row_count"],
                },
            )

        return AssetCheckResult(
            passed=not failed,
            metadata={
                "partition_count": len(partition_keys),
                "failed_partitions": MetadataValue.json(sorted(failed)),
                "passed_partition_count": len(partition_keys) - len(failed),
                "total_non_zero_volume_count": total_non_zero,
                "total_row_count": total_rows,
                "per_partition_counts": MetadataValue.json(per_partition_counts),
            },
        )
    finally:
        engine.dispose()


def _latest_materialized_row_count(
    instance, asset_key: AssetKey, partition_key: str
) -> int | None:
    """Return the `row_count` reported on the most recent AssetMaterialization
    event for `(asset_key, partition_key)` that carries `row_count` metadata,
    or `None` if no such event exists."""
    records = instance.fetch_materializations(
        AssetRecordsFilter(asset_key=asset_key, asset_partitions=[partition_key]),
        limit=10,
    ).records
    for record in records:
        materialization = record.asset_materialization
        if materialization is None:
            continue
        entry = materialization.metadata.get("row_count")
        if entry is None:
            continue
        value = entry.value if hasattr(entry, "value") else entry
        if value is None:
            continue
        return int(value)
    return None


@asset_check(
    asset=kraken_provider_asset_market_historical,
    name=PARTITION_DB_COUNT_AT_LEAST_MATERIALIZED_CHECK,
    description=(
        "For each materialized partition, verify the actual row count in "
        "provider_asset_market is at least the row_count reported in the "
        "latest materialization metadata. Catches silent upsert drops, "
        "constraint conflicts, or partial batch failures — the DB may have "
        "more rows (other backfills, overlapping providers) but never fewer "
        "than what we just claimed to upsert."
    ),
)
def check_partition_db_count_at_least_materialized(
    context: AssetCheckExecutionContext, postgres: PostgresResource
) -> AssetCheckResult:
    partition_keys = _resolve_check_partition_keys(context)
    asset_key = AssetKey("kraken_provider_asset_market_historical")

    engine = postgres.get_engine()
    try:
        with Session(engine) as session:
            provider_id = session.execute(
                select(Provider.id).where(Provider.name == "Kraken")
            ).scalar_one()

            per_partition: dict[str, dict[str, int | None]] = {}
            failed: list[str] = []
            for pk in partition_keys:
                materialized = _latest_materialized_row_count(
                    context.instance, asset_key, pk
                )
                actual = _count_partition_rows(
                    session, provider_id, dt.date.fromisoformat(pk)
                )
                per_partition[pk] = {
                    "materialized_row_count": materialized,
                    "db_row_count": actual,
                }
                if materialized is None:
                    failed.append(pk)
                    context.log.warning(
                        f"[check] partition={pk} no materialization metadata "
                        f"with row_count found; cannot verify db count"
                    )
                    continue
                passed = actual >= materialized
                if not passed:
                    failed.append(pk)
                context.log.info(
                    f"[check] partition={pk} "
                    f"materialized_row_count={materialized} "
                    f"db_row_count={actual} passed={passed}"
                )

        if len(partition_keys) == 1:
            pk = partition_keys[0]
            counts = per_partition[pk]
            return AssetCheckResult(
                passed=pk not in failed,
                metadata={
                    "partition": pk,
                    "materialized_row_count": counts["materialized_row_count"],
                    "db_row_count": counts["db_row_count"],
                },
            )

        return AssetCheckResult(
            passed=not failed,
            metadata={
                "partition_count": len(partition_keys),
                "failed_partitions": MetadataValue.json(sorted(failed)),
                "passed_partition_count": len(partition_keys) - len(failed),
                "per_partition_counts": MetadataValue.json(per_partition),
            },
        )
    finally:
        engine.dispose()


defs = Definitions(
    assets=[kraken_provider_asset_market_historical],
    asset_checks=[
        check_partition_has_non_zero_volume,
        check_partition_db_count_at_least_materialized,
    ],
)
