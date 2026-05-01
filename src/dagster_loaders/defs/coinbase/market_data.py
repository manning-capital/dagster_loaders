import time
import datetime as dt
from typing import Any, Final, Optional

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
    ScheduleDefinition,
    AssetExecutionContext,
    asset,
    asset_check,
    define_asset_job,
)
from mc_postgres_db.models import ProviderAssetMarket
from mc_postgres_db.operations import set_data

from dagster_loaders.resources import PostgresResource
from dagster_loaders.defs.data_quality import provider_market_data_quality
from dagster_loaders.defs.provider_assets import provider_asset_map

COINBASE_POOL: Final[str] = "coinbase-api"
COINBASE_RATE_LIMIT_SECONDS: Final[float] = 0.4
BATCH_SIZE: int = 5000
LOOKBACK_MINUTES: Final[int] = 60
GRANULARITY_SECONDS: Final[int] = 60
PRODUCTS_URL: Final[str] = "https://api.exchange.coinbase.com/products"
CANDLES_URL_TEMPLATE: Final[str] = (
    "https://api.exchange.coinbase.com/products/{product_id}/candles"
)


def _request_coinbase(url: str, params: Optional[dict[str, Any]] = None) -> Any:
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    return resp.json()


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
        "Pulls `GET /products` from the Coinbase Exchange API (filtered to "
        '`status == "online"`, not `trading_disabled`, not `auction_mode`), '
        "then `GET /products/{product_id}/candles` per product whose base + "
        "quote both map to active rows in `provider_asset` for the Coinbase "
        "provider. Window is the last 60 minutes at 60-second granularity. "
        "Rows are deduped on PK and upserted in batches of 5000 to stay "
        "under Postgres's bind-parameter cap."
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
        all_products = _request_coinbase(PRODUCTS_URL)
        products = [
            p
            for p in all_products
            if p.get("status") == "online"
            and not p.get("trading_disabled")
            and not p.get("auction_mode")
            and p.get("base_currency") in asset_map
            and p.get("quote_currency") in asset_map
        ]
        context.log.info(
            f"Coinbase returned {len(all_products)} products; "
            f"{len(products)} match status + asset map filters: "
            f"{[p['id'] for p in products]}"
        )

        end_ts = int(dt.datetime.now(dt.timezone.utc).timestamp())
        start_ts = end_ts - LOOKBACK_MINUTES * 60

        frames: list[pd.DataFrame] = []
        total = len(products)
        for idx, product in enumerate(products, start=1):
            product_id = product["id"]
            base_code = product["base_currency"]
            quote_code = product["quote_currency"]
            try:
                context.log.info(
                    f"[{idx}/{total}] Requesting candles for {product_id} "
                    f"({quote_code} -> {base_code})"
                )
                time.sleep(COINBASE_RATE_LIMIT_SECONDS)
                candles = _request_coinbase(
                    CANDLES_URL_TEMPLATE.format(product_id=product_id),
                    params={
                        "start": str(start_ts),
                        "end": str(end_ts),
                        "granularity": GRANULARITY_SECONDS,
                    },
                )
                if not candles:
                    context.log.info(
                        f"[{idx}/{total}] No candles returned for {product_id}"
                    )
                    continue
                df = pd.DataFrame(
                    candles,
                    columns=["time", "low", "high", "open", "close", "volume"],
                )
                df["timestamp"] = pd.to_datetime(df["time"].astype(int), unit="s")
                for c in ("open", "high", "low", "close", "volume"):
                    df[c] = df[c].astype(float)
                df = df.drop(columns=["time"])
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
        "every timestamp on a whole-minute boundary, and no per-pair density "
        "drop below 50% of the pair's own 30-day baseline. Scoped to the "
        "Coinbase provider only."
    ),
    blocking=False,
)
def coinbase_market_data_quality(postgres: PostgresResource) -> AssetCheckResult:
    engine = postgres.get_engine()
    try:
        return provider_market_data_quality(engine, "Coinbase")
    finally:
        engine.dispose()


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
