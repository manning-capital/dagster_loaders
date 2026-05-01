import datetime as dt
from typing import Any

import pytest
import responses
from dagster import AssetSelection, materialize
from sqlalchemy import Engine, select
from sqlalchemy.orm import Session
from mc_postgres_db.models import ProviderAssetMarket

from dagster_loaders.resources import PostgresResource
from dagster_loaders.defs.coinbase import market_data as coinbase_market_data
from dagster_loaders.defs.coinbase.market_data import (
    coinbase_market_data_quality,
    coinbase_provider_asset_market,
)


def _candle(ts: int, value: float = 100.0) -> list[float]:
    # Exchange API: [time, low, high, open, close, volume]
    return [ts, value, value, value, value, value]


def _stub_coinbase(
    products: list[dict[str, Any]],
    candles_by_product: dict[str, list[list[float]]],
) -> None:
    responses.add(
        responses.GET,
        coinbase_market_data.PRODUCTS_URL,
        json=products,
    )
    for product_id, candles in candles_by_product.items():
        responses.add(
            responses.GET,
            coinbase_market_data.CANDLES_URL_TEMPLATE.format(product_id=product_id),
            json=candles,
        )


def _online_product(
    product_id: str, base: str, quote: str, **overrides: Any
) -> dict[str, Any]:
    p = {
        "id": product_id,
        "base_currency": base,
        "quote_currency": quote,
        "status": "online",
        "trading_disabled": False,
        "auction_mode": False,
    }
    p.update(overrides)
    return p


def _materialize(engine: Engine):
    return materialize(
        [coinbase_provider_asset_market],
        resources={
            "postgres": PostgresResource(
                url=engine.url.render_as_string(hide_password=False)
            )
        },
    )


@responses.activate
def test_materialize_writes_rows_into_empty_db(
    postgres_engine: Engine, coinbase_base_data: dict[str, Any]
) -> None:
    ts = int(dt.datetime.now().replace(second=0, microsecond=0).timestamp())
    _stub_coinbase(
        products=[
            _online_product("BTC-USD", "BTC", "USD"),
            _online_product("ETH-USD", "ETH", "USD"),
        ],
        candles_by_product={
            "BTC-USD": [_candle(ts, "100")],
            "ETH-USD": [_candle(ts, "200")],
        },
    )

    result = _materialize(postgres_engine)
    assert result.success

    with Session(postgres_engine) as session:
        all_rows = session.execute(select(ProviderAssetMarket)).scalars().all()
        assert len(all_rows) == 2

        btc = session.execute(
            select(ProviderAssetMarket).where(
                ProviderAssetMarket.from_asset_id == coinbase_base_data["usd_asset_id"],
                ProviderAssetMarket.to_asset_id == coinbase_base_data["btc_asset_id"],
            )
        ).scalar_one()
        assert btc.open == 100.0
        assert btc.close == 100.0
        assert btc.volume == 100.0
        assert btc.provider_id == coinbase_base_data["provider_id"]

        eth = session.execute(
            select(ProviderAssetMarket).where(
                ProviderAssetMarket.from_asset_id == coinbase_base_data["usd_asset_id"],
                ProviderAssetMarket.to_asset_id == coinbase_base_data["eth_asset_id"],
            )
        ).scalar_one()
        assert eth.open == 200.0


@responses.activate
def test_upsert_overwrites_existing(
    postgres_engine: Engine, coinbase_base_data: dict[str, Any]
) -> None:
    ts = int(dt.datetime.now().replace(second=0, microsecond=0).timestamp())
    _stub_coinbase(
        products=[_online_product("BTC-USD", "BTC", "USD")],
        candles_by_product={"BTC-USD": [_candle(ts, "100")]},
    )
    assert _materialize(postgres_engine).success

    responses.reset()
    _stub_coinbase(
        products=[_online_product("BTC-USD", "BTC", "USD")],
        candles_by_product={"BTC-USD": [_candle(ts, "999")]},
    )
    assert _materialize(postgres_engine).success

    with Session(postgres_engine) as session:
        rows = session.execute(select(ProviderAssetMarket)).scalars().all()
        assert len(rows) == 1
        assert rows[0].close == 999.0
        assert rows[0].volume == 999.0


@responses.activate
def test_skips_products_not_in_asset_map(
    postgres_engine: Engine, coinbase_base_data: dict[str, Any]
) -> None:
    ts = int(dt.datetime.now().replace(second=0, microsecond=0).timestamp())
    _stub_coinbase(
        products=[
            _online_product("BTC-USD", "BTC", "USD"),
            _online_product("DOGE-USD", "DOGE", "USD"),
        ],
        candles_by_product={"BTC-USD": [_candle(ts, "100")]},
    )

    assert _materialize(postgres_engine).success

    with Session(postgres_engine) as session:
        rows = session.execute(select(ProviderAssetMarket)).scalars().all()
        assert len(rows) == 1
        assert rows[0].to_asset_id == coinbase_base_data["btc_asset_id"]


@pytest.mark.parametrize(
    "overrides",
    [
        {"status": "offline"},
        {"status": "delisted"},
        {"status": "internal"},
        {"trading_disabled": True},
        {"auction_mode": True},
    ],
)
@responses.activate
def test_filters_disabled_products(
    postgres_engine: Engine,
    coinbase_base_data: dict[str, Any],
    overrides: dict[str, Any],
) -> None:
    ts = int(dt.datetime.now().replace(second=0, microsecond=0).timestamp())
    _stub_coinbase(
        products=[
            _online_product("BTC-USD", "BTC", "USD"),
            _online_product("ETH-USD", "ETH", "USD", **overrides),
        ],
        candles_by_product={
            "BTC-USD": [_candle(ts, "100")],
        },
    )

    assert _materialize(postgres_engine).success

    with Session(postgres_engine) as session:
        rows = session.execute(select(ProviderAssetMarket)).scalars().all()
        assert len(rows) == 1
        assert rows[0].to_asset_id == coinbase_base_data["btc_asset_id"]


@pytest.mark.parametrize("batch_size", [1, 3, 100])
@responses.activate
def test_batches_large_dataset(
    monkeypatch: pytest.MonkeyPatch,
    postgres_engine: Engine,
    coinbase_base_data: dict[str, Any],
    batch_size: int,
) -> None:
    monkeypatch.setattr(coinbase_market_data, "BATCH_SIZE", batch_size)

    base_ts = int(dt.datetime.now().replace(second=0, microsecond=0).timestamp())
    n_per_product = 5
    _stub_coinbase(
        products=[
            _online_product("BTC-USD", "BTC", "USD"),
            _online_product("ETH-USD", "ETH", "USD"),
        ],
        candles_by_product={
            "BTC-USD": [_candle(base_ts + 60 * i, "1") for i in range(n_per_product)],
            "ETH-USD": [_candle(base_ts + 60 * i, "2") for i in range(n_per_product)],
        },
    )

    assert _materialize(postgres_engine).success

    with Session(postgres_engine) as session:
        rows = session.execute(select(ProviderAssetMarket)).scalars().all()
        assert len(rows) == n_per_product * 2


def _run_check_only(engine: Engine):
    return materialize(
        [coinbase_provider_asset_market, coinbase_market_data_quality],
        selection=AssetSelection.checks(coinbase_market_data_quality),
        resources={
            "postgres": PostgresResource(
                url=engine.url.render_as_string(hide_password=False)
            )
        },
    )


def _seed_market_rows(
    engine: Engine,
    ids: dict[str, Any],
    timestamps: list[dt.datetime],
    close: float = 100.0,
) -> None:
    with Session(engine) as session:
        session.add_all(
            [
                ProviderAssetMarket(
                    timestamp=ts,
                    provider_id=ids["provider_id"],
                    from_asset_id=ids["usd_asset_id"],
                    to_asset_id=ids["btc_asset_id"],
                    open=close,
                    high=close,
                    low=close,
                    close=close,
                    volume=1.0,
                )
                for ts in timestamps
            ]
        )
        session.commit()


@responses.activate
def test_data_quality_check_passes_after_materialize(
    postgres_engine: Engine, coinbase_base_data: dict[str, Any]
) -> None:
    now = dt.datetime.now(dt.timezone.utc).replace(second=0, microsecond=0)
    ts = int(now.timestamp())
    _stub_coinbase(
        products=[_online_product("BTC-USD", "BTC", "USD")],
        candles_by_product={"BTC-USD": [_candle(ts, "100")]},
    )
    assert _materialize(postgres_engine).success

    result = _run_check_only(postgres_engine)
    assert result.success
    evals = result.get_asset_check_evaluations()
    assert len(evals) == 1
    assert evals[0].passed is True
    assert evals[0].metadata["rows_in_last_2h"].value == 1
    assert evals[0].metadata["off_minute_rows"].value == 0
    assert evals[0].metadata["low_density_pairs_count"].value == 0
    assert evals[0].metadata["min_close_price"].value == 100.0


def test_data_quality_check_fails_when_table_empty(
    postgres_engine: Engine, coinbase_base_data: dict[str, Any]
) -> None:
    result = _run_check_only(postgres_engine)
    assert result.success
    evals = result.get_asset_check_evaluations()
    assert len(evals) == 1
    assert evals[0].passed is False
    assert "no rows" in (evals[0].description or "")


def test_data_quality_check_fails_on_off_minute_row(
    postgres_engine: Engine, coinbase_base_data: dict[str, Any]
) -> None:
    base = dt.datetime.now(dt.timezone.utc).replace(
        second=0, microsecond=0, tzinfo=None
    )
    _seed_market_rows(
        postgres_engine,
        coinbase_base_data,
        [
            base,
            base + dt.timedelta(seconds=37),
            base + dt.timedelta(minutes=2),
        ],
    )

    result = _run_check_only(postgres_engine)
    evals = result.get_asset_check_evaluations()
    assert evals[0].passed is False
    assert evals[0].metadata["off_minute_rows"].value == 1
    assert "whole-minute" in (evals[0].description or "")


def test_check_ignores_kraken_rows(
    postgres_engine: Engine,
    coinbase_base_data: dict[str, Any],
    kraken_base_data: dict[str, Any],
) -> None:
    """Kraken rows in the table must not affect the Coinbase check."""
    base = dt.datetime.now(dt.timezone.utc).replace(
        second=0, microsecond=0, tzinfo=None
    )
    with Session(postgres_engine) as session:
        session.add_all(
            [
                ProviderAssetMarket(
                    timestamp=base,
                    provider_id=kraken_base_data["provider_id"],
                    from_asset_id=kraken_base_data["usd_asset_id"],
                    to_asset_id=kraken_base_data["btc_asset_id"],
                    open=100.0,
                    high=100.0,
                    low=100.0,
                    close=100.0,
                    volume=1.0,
                ),
            ]
        )
        session.commit()

    result = _run_check_only(postgres_engine)
    evals = result.get_asset_check_evaluations()
    assert evals[0].passed is False
    assert evals[0].metadata["rows_in_last_2h"].value == 0
    assert "no rows" in (evals[0].description or "")
