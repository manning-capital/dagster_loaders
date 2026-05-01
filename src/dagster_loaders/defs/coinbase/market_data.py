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
from sqlalchemy import func, select
from sqlalchemy.orm import Session
from mc_postgres_db.models import Provider, ProviderAssetMarket
from mc_postgres_db.operations import set_data

from dagster_loaders.resources import PostgresResource
from dagster_loaders.defs.provider_assets import provider_asset_map

COINBASE_POOL = "coinbase-api"
COINBASE_RATE_LIMIT_SECONDS = 0.4
BATCH_SIZE = 5000
LOOKBACK_MINUTES = 60
GRANULARITY = "ONE_MINUTE"
PRODUCTS_URL = "https://api.coinbase.com/api/v3/brokerage/market/products"
CANDLES_URL_TEMPLATE = (
    "https://api.coinbase.com/api/v3/brokerage/market/products/{product_id}/candles"
)


def _request_coinbase(
    url: str, params: Optional[dict[str, Any]] = None
) -> dict[str, Any]:
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    body = resp.json()
    if isinstance(body, dict) and body.get("error"):
        raise RuntimeError(f"Coinbase {url} returned error: {body['error']}")
    return body


@asset(
    pool=COINBASE_POOL,
    group_name="market_data",
    kinds={"python", "postgres"},
    owners=["glynfinck@gmail.com"],
    tags={"domain": "market-data", "provider": "coinbase"},
    retry_policy=RetryPolicy(max_retries=3, delay=5.0, backoff=Backoff.EXPONENTIAL),
    freshness_policy=FreshnessPolicy.time_window(
        fail_window=dt.timedelta(minutes=60),
        warn_window=dt.timedelta(minutes=45),
    ),
    description=(
        "Coinbase OHLCV candles upserted into `provider_asset_market`.\n\n"
        "Pulls `GET /api/v3/brokerage/market/products` (filtered to online, "
        "non-disabled spot products), then `GET "
        "/api/v3/brokerage/market/products/{product_id}/candles` per product "
        "whose base+quote both map to active rows in `provider_asset` for "
        "the Coinbase provider. Window is the last 60 minutes at "
        "ONE_MINUTE granularity. Rows are deduped on PK and upserted in "
        "batches of 5000 to stay under Postgres's bind-parameter cap."
    ),
)
def coinbase_provider_asset_market(
    context: AssetExecutionContext, postgres: PostgresResource
) -> MaterializeResult:
    as_of = dt.date.today()
    engine = postgres.get_engine()
    skipped_products: list[str] = []
    try:
        provider_id, asset_map = provider_asset_map(engine, "Coinbase", as_of)
        context.log.info(f"Loaded {len(asset_map)} Coinbase provider assets")

        time.sleep(COINBASE_RATE_LIMIT_SECONDS)
        products_body = _request_coinbase(PRODUCTS_URL)
        all_products = products_body.get("products", [])
        products = [
            p
            for p in all_products
            if p.get("status") == "online"
            and not p.get("is_disabled")
            and not p.get("trading_disabled")
            and not p.get("view_only")
            and not p.get("auction_mode")
            and p.get("base_currency_id") in asset_map
            and p.get("quote_currency_id") in asset_map
        ]
        context.log.info(
            f"Coinbase returned {len(all_products)} products; "
            f"{len(products)} match status + asset map filters: "
            f"{[p['product_id'] for p in products]}"
        )

        end_ts = int(dt.datetime.now(dt.timezone.utc).timestamp())
        start_ts = end_ts - LOOKBACK_MINUTES * 60

        frames: list[pd.DataFrame] = []
        total = len(products)
        for idx, product in enumerate(products, start=1):
            product_id = product["product_id"]
            base_code = product["base_currency_id"]
            quote_code = product["quote_currency_id"]
            try:
                context.log.info(
                    f"[{idx}/{total}] Requesting candles for {product_id} "
                    f"({quote_code} -> {base_code})"
                )
                time.sleep(COINBASE_RATE_LIMIT_SECONDS)
                candles_body = _request_coinbase(
                    CANDLES_URL_TEMPLATE.format(product_id=product_id),
                    params={
                        "start": str(start_ts),
                        "end": str(end_ts),
                        "granularity": GRANULARITY,
                    },
                )
                candles = candles_body.get("candles", [])
                if not candles:
                    context.log.info(
                        f"[{idx}/{total}] No candles returned for {product_id}"
                    )
                    continue
                df = pd.DataFrame(candles)
                df["timestamp"] = pd.to_datetime(df["start"].astype(int), unit="s")
                for c in ("open", "high", "low", "close", "volume"):
                    df[c] = df[c].astype(float)
                df = df.drop(columns=["start"])
                df["from_asset_id"] = asset_map[quote_code]
                df["to_asset_id"] = asset_map[base_code]
                df["provider_id"] = provider_id
                frames.append(df)
                context.log.info(
                    f"[{idx}/{total}] Fetched {len(df)} rows for {product_id}"
                )
            except Exception as e:
                context.log.error(f"[{idx}/{total}] Skipping product {product_id}: {e}")
                skipped_products.append(product_id)

        if not frames:
            context.log.info("No rows to write.")
            return MaterializeResult(
                metadata={
                    "row_count": 0,
                    "product_count": len(products),
                    "skipped_products": MetadataValue.json(skipped_products),
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
                "product_count": len(products),
                "skipped_products": MetadataValue.json(skipped_products),
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
    asset=coinbase_provider_asset_market,
    name="coinbase_market_data_quality",
    description=(
        "Within the recent 2h window: rows present, all close prices > 0, "
        "every timestamp on a whole-minute boundary, and per-pair points "
        "uniformly spaced inside [min, max] (edges ignored). Scoped to "
        "the Coinbase provider only."
    ),
    blocking=False,
)
def coinbase_market_data_quality(postgres: PostgresResource) -> AssetCheckResult:
    engine = postgres.get_engine()
    try:
        with Session(engine) as session:
            provider_id = session.execute(
                select(Provider.id).where(Provider.name == "Coinbase")
            ).scalar_one()

            recent_cutoff = dt.datetime.now(dt.timezone.utc).replace(
                tzinfo=None
            ) - dt.timedelta(hours=2)

            recent_rows = session.execute(
                select(func.count())
                .select_from(ProviderAssetMarket)
                .where(
                    ProviderAssetMarket.provider_id == provider_id,
                    ProviderAssetMarket.timestamp >= recent_cutoff,
                )
            ).scalar_one()

            min_close = session.execute(
                select(func.min(ProviderAssetMarket.close)).where(
                    ProviderAssetMarket.provider_id == provider_id,
                    ProviderAssetMarket.timestamp >= recent_cutoff,
                )
            ).scalar_one()

            off_minute_rows = session.execute(
                select(func.count())
                .select_from(ProviderAssetMarket)
                .where(
                    ProviderAssetMarket.provider_id == provider_id,
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
                .where(
                    ProviderAssetMarket.provider_id == provider_id,
                    ProviderAssetMarket.timestamp >= recent_cutoff,
                )
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


coinbase_market_job = define_asset_job(
    name="coinbase_market_job",
    selection=[coinbase_provider_asset_market],
)

coinbase_market_schedule = ScheduleDefinition(
    name="coinbase_market_every_30min",
    cron_schedule="*/30 * * * *",
    job=coinbase_market_job,
)


defs = Definitions(
    assets=[coinbase_provider_asset_market],
    asset_checks=[coinbase_market_data_quality],
    jobs=[coinbase_market_job],
    schedules=[coinbase_market_schedule],
)
