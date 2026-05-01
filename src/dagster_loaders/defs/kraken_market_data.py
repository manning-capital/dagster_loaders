import time
import datetime as dt
from typing import Any, Optional

import pandas as pd
import requests
from dagster import (
    Backoff,
    Definitions,
    RetryPolicy,
    MetadataValue,
    FreshnessPolicy,
    AssetCheckResult,
    MaterializeResult,
    AssetCheckSeverity,
    ScheduleDefinition,
    AssetExecutionContext,
    asset,
    asset_check,
    define_asset_job,
)
from sqlalchemy import Engine, func, select
from sqlalchemy.orm import Session
from mc_postgres_db.models import Asset, Provider, ProviderAsset, ProviderAssetMarket
from mc_postgres_db.operations import set_data

from dagster_loaders.resources import PostgresResource

KRAKEN_POOL = "kraken-api"
KRAKEN_RATE_LIMIT_SECONDS = 1.0
BATCH_SIZE = 5000
ASSET_PAIRS_URL = "https://api.kraken.com/0/public/AssetPairs"
OHLC_URL = "https://api.kraken.com/0/public/OHLC"


def _request_kraken(
    url: str, params: Optional[dict[str, Any]] = None
) -> dict[str, Any]:
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    body = resp.json()
    errors = body.get("error") or []
    if errors:
        raise RuntimeError(f"Kraken {url} returned errors: {errors}")
    return body


def _kraken_provider_asset_map(
    engine: Engine, as_of: dt.date
) -> tuple[int, dict[str, int]]:
    with Session(engine) as session:
        provider_id = session.execute(
            select(Provider.id).where(Provider.name == "Kraken")
        ).scalar_one()

        subq = (
            select(
                ProviderAsset.asset_code,
                ProviderAsset.provider_id,
                func.max(ProviderAsset.date).label("max_date"),
            )
            .where(ProviderAsset.date <= as_of, ProviderAsset.is_active.is_(True))
            .group_by(ProviderAsset.asset_code, ProviderAsset.provider_id)
            .subquery()
        )
        q = (
            select(ProviderAsset.asset_code, ProviderAsset.asset_id)
            .join(
                subq,
                (ProviderAsset.asset_code == subq.c.asset_code)
                & (ProviderAsset.provider_id == subq.c.provider_id)
                & (ProviderAsset.date == subq.c.max_date),
            )
            .join(Asset, Asset.id == ProviderAsset.asset_id)
            .where(ProviderAsset.provider_id == provider_id, Asset.is_active.is_(True))
        )
        rows = session.execute(q).all()

    return provider_id, {code: aid for code, aid in rows}


@asset(
    pool=KRAKEN_POOL,
    group_name="market_data",
    kinds={"python", "postgres"},
    owners=["glynfinck@gmail.com"],
    tags={"domain": "market-data", "provider": "kraken"},
    retry_policy=RetryPolicy(max_retries=3, delay=5.0, backoff=Backoff.EXPONENTIAL),
    freshness_policy=FreshnessPolicy.time_window(
        fail_window=dt.timedelta(minutes=60),
        warn_window=dt.timedelta(minutes=45),
    ),
    description=(
        "Kraken OHLCV candles upserted into `provider_asset_market`.\n\n"
        "Pulls `GET /0/public/AssetPairs` (filtered to `execution_venue == "
        '"international"`), then `GET /0/public/OHLC` per pair whose '
        "base+quote both map to active rows in `provider_asset` for the "
        "Kraken provider. Rows are deduped on PK and upserted in batches "
        "of 5000 to stay under Postgres's bind-parameter cap."
    ),
)
def kraken_provider_asset_market(
    context: AssetExecutionContext, postgres: PostgresResource
) -> MaterializeResult:
    as_of = dt.date.today()
    engine = postgres.get_engine()
    skipped_pairs: list[str] = []
    try:
        provider_id, asset_map = _kraken_provider_asset_map(engine, as_of)
        context.log.info(f"Loaded {len(asset_map)} Kraken provider assets")

        time.sleep(KRAKEN_RATE_LIMIT_SECONDS)
        pairs_body = _request_kraken(ASSET_PAIRS_URL)
        all_pairs_count = len(pairs_body["result"])
        pairs = [
            (code, info["quote"], info["base"])
            for code, info in pairs_body["result"].items()
            if info.get("execution_venue", "international") == "international"
            and info["quote"] in asset_map
            and info["base"] in asset_map
        ]
        context.log.info(
            f"Kraken returned {all_pairs_count} pairs; "
            f"{len(pairs)} match venue + asset map filters: "
            f"{[code for code, _, _ in pairs]}"
        )

        frames: list[pd.DataFrame] = []
        total = len(pairs)
        for idx, (code, from_code, to_code) in enumerate(pairs, start=1):
            try:
                context.log.info(
                    f"[{idx}/{total}] Requesting OHLC for {code} ({from_code} -> {to_code})"
                )
                time.sleep(KRAKEN_RATE_LIMIT_SECONDS)
                ohlc_body = _request_kraken(OHLC_URL, params={"pair": code})
                rows = ohlc_body["result"][code]
                df = pd.DataFrame(
                    rows,
                    columns=[
                        "timestamp",
                        "open",
                        "high",
                        "low",
                        "close",
                        "vwap",
                        "volume",
                        "count",
                    ],
                )
                df = df.drop(columns=["vwap", "count"])
                df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
                for c in ("open", "high", "low", "close", "volume"):
                    df[c] = df[c].astype(float)
                df["from_asset_id"] = asset_map[from_code]
                df["to_asset_id"] = asset_map[to_code]
                df["provider_id"] = provider_id
                frames.append(df)
                context.log.info(f"[{idx}/{total}] Fetched {len(df)} rows for {code}")
            except Exception as e:
                context.log.error(f"[{idx}/{total}] Skipping pair {code}: {e}")
                skipped_pairs.append(code)

        if not frames:
            context.log.info("No rows to write.")
            return MaterializeResult(
                metadata={
                    "row_count": 0,
                    "pair_count": len(pairs),
                    "skipped_pairs": MetadataValue.json(skipped_pairs),
                    "table": ProviderAssetMarket.__tablename__,
                    "as_of": MetadataValue.text(as_of.isoformat()),
                }
            )

        data = pd.concat(frames, ignore_index=True)
        data = data.drop_duplicates(
            subset=["timestamp", "provider_id", "from_asset_id", "to_asset_id"],
            keep="last",
        )

        for i in range(0, len(data), BATCH_SIZE):
            batch = data.iloc[i : i + BATCH_SIZE]
            set_data(engine, ProviderAssetMarket.__tablename__, batch, "upsert")
        context.log.info(
            f"Upserted {len(data)} rows into {ProviderAssetMarket.__tablename__}"
        )

        return MaterializeResult(
            metadata={
                "row_count": len(data),
                "pair_count": len(pairs),
                "skipped_pairs": MetadataValue.json(skipped_pairs),
                "table": ProviderAssetMarket.__tablename__,
                "as_of": MetadataValue.text(as_of.isoformat()),
                "min_timestamp": MetadataValue.text(
                    data["timestamp"].min().isoformat()
                ),
                "max_timestamp": MetadataValue.text(
                    data["timestamp"].max().isoformat()
                ),
                "preview": MetadataValue.md(data.head().to_markdown(index=False)),
            }
        )
    finally:
        engine.dispose()


@asset_check(
    asset=kraken_provider_asset_market,
    name="kraken_market_data_quality",
    description=(
        "Within the recent 2h window: rows present, all close prices > 0, "
        "every timestamp on a whole-minute boundary, and per-pair points "
        "uniformly spaced inside [min, max] (edges ignored)."
    ),
    blocking=False,
)
def kraken_market_data_quality(postgres: PostgresResource) -> AssetCheckResult:
    engine = postgres.get_engine()
    try:
        with Session(engine) as session:
            recent_cutoff = dt.datetime.now(dt.timezone.utc).replace(
                tzinfo=None
            ) - dt.timedelta(hours=2)

            recent_rows = session.execute(
                select(func.count())
                .select_from(ProviderAssetMarket)
                .where(ProviderAssetMarket.timestamp >= recent_cutoff)
            ).scalar_one()

            min_close = session.execute(
                select(func.min(ProviderAssetMarket.close)).where(
                    ProviderAssetMarket.timestamp >= recent_cutoff
                )
            ).scalar_one()

            off_minute_rows = session.execute(
                select(func.count())
                .select_from(ProviderAssetMarket)
                .where(
                    ProviderAssetMarket.timestamp >= recent_cutoff,
                    func.extract("second", ProviderAssetMarket.timestamp) != 0,
                )
            ).scalar_one()

            pair_stats = session.execute(
                select(
                    ProviderAssetMarket.from_asset_id,
                    ProviderAssetMarket.to_asset_id,
                    ProviderAssetMarket.provider_id,
                    func.min(ProviderAssetMarket.timestamp).label("min_ts"),
                    func.max(ProviderAssetMarket.timestamp).label("max_ts"),
                    func.count().label("actual"),
                )
                .where(ProviderAssetMarket.timestamp >= recent_cutoff)
                .group_by(
                    ProviderAssetMarket.from_asset_id,
                    ProviderAssetMarket.to_asset_id,
                    ProviderAssetMarket.provider_id,
                )
            ).all()
    finally:
        engine.dispose()

    gappy_pairs: list[dict[str, Any]] = []
    for row in pair_stats:
        if row.actual <= 1:
            continue
        expected = int((row.max_ts - row.min_ts).total_seconds() // 60) + 1
        if row.actual < expected:
            gappy_pairs.append(
                {
                    "from_asset_id": row.from_asset_id,
                    "to_asset_id": row.to_asset_id,
                    "provider_id": row.provider_id,
                    "actual": row.actual,
                    "expected": expected,
                    "missing": expected - row.actual,
                }
            )

    failures: list[str] = []
    if recent_rows == 0:
        failures.append("no rows with timestamp in the last 2h")
    if min_close is not None and min_close <= 0:
        failures.append(f"min(close) = {min_close} (expected > 0)")
    if off_minute_rows > 0:
        failures.append(
            f"{off_minute_rows} rows are not aligned to a whole-minute boundary"
        )
    if gappy_pairs:
        failures.append(
            f"{len(gappy_pairs)} pair(s) have minute gaps inside the recent window"
        )

    return AssetCheckResult(
        passed=not failures,
        severity=AssetCheckSeverity.WARN,
        description="; ".join(failures) if failures else "ok",
        metadata={
            "rows_in_last_2h": recent_rows,
            "min_close_price": MetadataValue.float(float(min_close or 0.0)),
            "off_minute_rows": off_minute_rows,
            "gappy_pairs_count": len(gappy_pairs),
            "gappy_pairs": MetadataValue.json(gappy_pairs),
        },
    )


kraken_market_job = define_asset_job(
    name="kraken_market_job",
    selection=[kraken_provider_asset_market],
)

kraken_market_schedule = ScheduleDefinition(
    name="kraken_market_every_30min",
    cron_schedule="*/30 * * * *",
    job=kraken_market_job,
)


defs = Definitions(
    assets=[kraken_provider_asset_market],
    asset_checks=[kraken_market_data_quality],
    jobs=[kraken_market_job],
    schedules=[kraken_market_schedule],
)
