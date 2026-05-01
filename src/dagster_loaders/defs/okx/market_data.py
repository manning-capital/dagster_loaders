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
    AssetExecutionContext,
    asset,
    asset_check,
)
from mc_postgres_db.models import ProviderAssetMarket
from mc_postgres_db.operations import set_data

from dagster_loaders.resources import PostgresResource
from dagster_loaders.defs.data_quality import provider_market_data_quality
from dagster_loaders.defs.provider_assets import provider_asset_map

OKX_POOL: Final[str] = "okx-api"
OKX_RATE_LIMIT_SECONDS: Final[float] = 0.5
BATCH_SIZE: int = 5000
LOOKBACK_MINUTES: Final[int] = 60
BAR: Final[str] = "1m"
INSTRUMENTS_URL: Final[str] = "https://www.okx.com/api/v5/public/instruments"
CANDLES_URL: Final[str] = "https://www.okx.com/api/v5/market/candles"


def _request_okx(url: str, params: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    body = resp.json()
    if str(body.get("code")) != "0":
        raise RuntimeError(
            f"OKX {url} returned code={body.get('code')} msg={body.get('msg')!r}"
        )
    return body


@asset(
    pool=OKX_POOL,
    group_name="market_data",
    kinds={"python", "postgres"},
    owners=["glynfinck@gmail.com"],
    tags={"domain": "market-data", "provider": "okx"},
    retry_policy=RetryPolicy(max_retries=3, delay=5.0, backoff=Backoff.EXPONENTIAL),
    freshness_policy=FreshnessPolicy.time_window(
        fail_window=dt.timedelta(minutes=60),
        warn_window=dt.timedelta(minutes=45),
    ),
    description=(
        "OKX OHLCV candles upserted into `provider_asset_market`.\n\n"
        "Pulls `GET /api/v5/public/instruments?instType=SPOT` (filtered to "
        '`state == "live"`), then `GET /api/v5/market/candles?bar=1m` per '
        "instrument whose base+quote both map to active rows in "
        "`provider_asset` for the OKX provider. Drops the in-progress current "
        'candle (`confirm == "0"`) so all written timestamps land on whole-'
        "minute boundaries. Window is the last 60 minutes. Rows are deduped "
        "on PK and upserted in batches of 5000 to stay under Postgres's "
        "bind-parameter cap."
    ),
)
def okx_provider_asset_market(
    context: AssetExecutionContext, postgres: PostgresResource
) -> MaterializeResult:
    as_of = dt.date.today()
    engine = postgres.get_engine()
    skipped_instruments: list[str] = []
    try:
        provider_id, asset_map = provider_asset_map(engine, "OKX", as_of)
        context.log.info(f"Loaded {len(asset_map)} OKX provider assets")

        time.sleep(OKX_RATE_LIMIT_SECONDS)
        instruments_body = _request_okx(INSTRUMENTS_URL, params={"instType": "SPOT"})
        all_instruments = instruments_body["data"]
        instruments = [
            i
            for i in all_instruments
            if i.get("state") == "live"
            and i.get("baseCcy") in asset_map
            and i.get("quoteCcy") in asset_map
        ]
        context.log.info(
            f"OKX returned {len(all_instruments)} instruments; "
            f"{len(instruments)} match state + asset map filters: "
            f"{[i['instId'] for i in instruments]}"
        )

        frames: list[pd.DataFrame] = []
        total = len(instruments)
        for idx, instrument in enumerate(instruments, start=1):
            inst_id = instrument["instId"]
            base_code = instrument["baseCcy"]
            quote_code = instrument["quoteCcy"]
            try:
                context.log.info(
                    f"[{idx}/{total}] Requesting candles for {inst_id} "
                    f"({quote_code} -> {base_code})"
                )
                time.sleep(OKX_RATE_LIMIT_SECONDS)
                candles_body = _request_okx(
                    CANDLES_URL,
                    params={
                        "instId": inst_id,
                        "bar": BAR,
                        "limit": str(LOOKBACK_MINUTES),
                    },
                )
                rows = candles_body["data"]
                if not rows:
                    context.log.info(
                        f"[{idx}/{total}] No candles returned for {inst_id}"
                    )
                    continue
                # OKX rows: [ts, o, h, l, c, vol, volCcy, volCcyQuote, confirm].
                # Slice by index so a future field append doesn't break parsing.
                df = pd.DataFrame(
                    [
                        {
                            "ts_ms": r[0],
                            "open": r[1],
                            "high": r[2],
                            "low": r[3],
                            "close": r[4],
                            "volume": r[5],
                            "confirm": r[8],
                        }
                        for r in rows
                    ]
                )
                # Drop the in-progress current minute; only keep closed candles.
                df = df[df["confirm"] == "1"]
                if df.empty:
                    context.log.info(
                        f"[{idx}/{total}] All candles for {inst_id} were unconfirmed"
                    )
                    continue
                df["timestamp"] = pd.to_datetime(
                    df["ts_ms"].astype("int64") // 1000, unit="s"
                )
                for c in ("open", "high", "low", "close", "volume"):
                    df[c] = df[c].astype(float)
                df = df.drop(columns=["ts_ms", "confirm"])
                # Match the Coinbase orientation: from = quote, to = base.
                df["from_asset_id"] = asset_map[quote_code]
                df["to_asset_id"] = asset_map[base_code]
                df["provider_id"] = provider_id
                frames.append(df)
                context.log.info(
                    f"[{idx}/{total}] Fetched {len(df)} rows for {inst_id}"
                )
            except Exception as e:
                context.log.error(f"[{idx}/{total}] Skipping instrument {inst_id}: {e}")
                skipped_instruments.append(inst_id)

        if not frames:
            context.log.info("No rows to write.")
            return MaterializeResult(
                metadata={
                    "row_count": 0,
                    "instrument_count": len(instruments),
                    "skipped_instruments": MetadataValue.json(skipped_instruments),
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
                "instrument_count": len(instruments),
                "skipped_instruments": MetadataValue.json(skipped_instruments),
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
    asset=okx_provider_asset_market,
    name="okx_market_data_quality",
    description=(
        "Within the recent 2h window: rows present, all close prices > 0, "
        "every timestamp on a whole-minute boundary, and no per-pair density "
        "drop below 50% of the pair's own 30-day baseline. Scoped to the "
        "OKX provider only."
    ),
    blocking=False,
)
def okx_market_data_quality(postgres: PostgresResource) -> AssetCheckResult:
    engine = postgres.get_engine()
    try:
        return provider_market_data_quality(engine, "OKX")
    finally:
        engine.dispose()


defs = Definitions(
    assets=[okx_provider_asset_market],
    asset_checks=[okx_market_data_quality],
)
