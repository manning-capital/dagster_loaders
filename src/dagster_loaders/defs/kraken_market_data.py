import datetime as dt
import time
from typing import Any, Optional

import pandas as pd
import requests
from dagster import (
    AssetExecutionContext,
    ConfigurableResource,
    Definitions,
    EnvVar,
    ScheduleDefinition,
    asset,
    define_asset_job,
)
from mc_postgres_db.models import Asset, Provider, ProviderAsset, ProviderAssetMarket
from mc_postgres_db.operations import set_data
from sqlalchemy import Engine, create_engine, func, select
from sqlalchemy.orm import Session


KRAKEN_POOL = "kraken-api"
KRAKEN_RATE_LIMIT_SECONDS = 1.0
BATCH_SIZE = 5000
ASSET_PAIRS_URL = "https://api.kraken.com/0/public/AssetPairs"
OHLC_URL = "https://api.kraken.com/0/public/OHLC"


class PostgresResource(ConfigurableResource):
    url: str

    def get_engine(self) -> Engine:
        return create_engine(self.url)


def _request_kraken(url: str, params: Optional[dict[str, Any]] = None) -> dict[str, Any]:
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
)
def kraken_provider_asset_market(
    context: AssetExecutionContext, postgres: PostgresResource
) -> None:
    as_of = dt.date.today()
    engine = postgres.get_engine()
    try:
        provider_id, asset_map = _kraken_provider_asset_map(engine, as_of)
        context.log.info(f"Loaded {len(asset_map)} Kraken provider assets")

        time.sleep(KRAKEN_RATE_LIMIT_SECONDS)
        pairs_body = _request_kraken(ASSET_PAIRS_URL)
        pairs = [
            (code, info["quote"], info["base"])
            for code, info in pairs_body["result"].items()
            if info.get("execution_venue", "international") == "international"
            and info["quote"] in asset_map
            and info["base"] in asset_map
        ]
        context.log.info(f"Kraken pairs after filtering: {len(pairs)}")

        frames: list[pd.DataFrame] = []
        for code, from_code, to_code in pairs:
            try:
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
            except Exception as e:
                context.log.error(f"Skipping pair {code}: {e}")

        if not frames:
            context.log.info("No rows to write.")
            return

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
    finally:
        engine.dispose()


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
    jobs=[kraken_market_job],
    schedules=[kraken_market_schedule],
    resources={"postgres": PostgresResource(url=EnvVar("POSTGRES_URL"))},
)
