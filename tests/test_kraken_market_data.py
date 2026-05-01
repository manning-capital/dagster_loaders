import datetime as dt
from typing import Any

import pytest
import responses
from dagster import AssetSelection, materialize
from sqlalchemy import Engine, select
from sqlalchemy.orm import Session
from mc_postgres_db.models import ProviderAssetMarket

from dagster_loaders.resources import PostgresResource
from dagster_loaders.defs.kraken import market_data as kraken_market_data
from dagster_loaders.defs.kraken.market_data import (
    kraken_market_data_quality,
    kraken_provider_asset_market,
)


def _stub_kraken(
    asset_pairs: dict[str, dict[str, Any]],
    market_data: dict[str, list[list[Any]]],
) -> None:
    responses.add(
        responses.GET,
        kraken_market_data.ASSET_PAIRS_URL,
        json={"result": asset_pairs},
    )
    for pair_code, rows in market_data.items():
        responses.add(
            responses.GET,
            kraken_market_data.OHLC_URL,
            json={"result": {pair_code: rows}, "error": []},
            match=[
                responses.matchers.query_param_matcher(
                    {"pair": pair_code}, strict_match=False
                )
            ],
        )


def _materialize(engine: Engine):
    return materialize(
        [kraken_provider_asset_market],
        resources={
            "postgres": PostgresResource(
                url=engine.url.render_as_string(hide_password=False)
            )
        },
    )


@responses.activate
def test_materialize_writes_rows_into_empty_db(
    postgres_engine: Engine, kraken_base_data: dict[str, Any]
) -> None:
    ts = int(dt.datetime.now().timestamp())
    _stub_kraken(
        asset_pairs={
            "XXBTZUSD": {"base": "XXBT", "quote": "ZUSD"},
            "XETHZUSD": {"base": "XETH", "quote": "ZUSD"},
        },
        market_data={
            "XXBTZUSD": [[ts, "100", "100", "100", "100", "100", "100", 100]],
            "XETHZUSD": [[ts, "200", "200", "200", "200", "200", "200", 100]],
        },
    )

    result = _materialize(postgres_engine)
    assert result.success

    with Session(postgres_engine) as session:
        all_rows = session.execute(select(ProviderAssetMarket)).scalars().all()
        assert len(all_rows) == 2

        btc = session.execute(
            select(ProviderAssetMarket).where(
                ProviderAssetMarket.from_asset_id == kraken_base_data["usd_asset_id"],
                ProviderAssetMarket.to_asset_id == kraken_base_data["btc_asset_id"],
            )
        ).scalar_one()
        assert btc.open == 100.0
        assert btc.close == 100.0
        assert btc.volume == 100.0
        assert btc.provider_id == kraken_base_data["provider_id"]

        eth = session.execute(
            select(ProviderAssetMarket).where(
                ProviderAssetMarket.from_asset_id == kraken_base_data["usd_asset_id"],
                ProviderAssetMarket.to_asset_id == kraken_base_data["eth_asset_id"],
            )
        ).scalar_one()
        assert eth.open == 200.0


@responses.activate
def test_upsert_overwrites_existing(
    postgres_engine: Engine, kraken_base_data: dict[str, Any]
) -> None:
    ts = int(dt.datetime.now().timestamp())
    _stub_kraken(
        asset_pairs={"XXBTZUSD": {"base": "XXBT", "quote": "ZUSD"}},
        market_data={
            "XXBTZUSD": [[ts, "100", "100", "100", "100", "100", "100", 1]],
        },
    )
    assert _materialize(postgres_engine).success

    responses.reset()
    _stub_kraken(
        asset_pairs={"XXBTZUSD": {"base": "XXBT", "quote": "ZUSD"}},
        market_data={
            "XXBTZUSD": [[ts, "999", "999", "999", "999", "999", "999", 1]],
        },
    )
    assert _materialize(postgres_engine).success

    with Session(postgres_engine) as session:
        rows = session.execute(select(ProviderAssetMarket)).scalars().all()
        assert len(rows) == 1
        assert rows[0].close == 999.0
        assert rows[0].volume == 999.0


@responses.activate
def test_skips_pairs_not_in_asset_map(
    postgres_engine: Engine, kraken_base_data: dict[str, Any]
) -> None:
    ts = int(dt.datetime.now().timestamp())
    _stub_kraken(
        asset_pairs={
            "XXBTZUSD": {"base": "XXBT", "quote": "ZUSD"},
            "DOGEZUSD": {"base": "DOGE", "quote": "ZUSD"},
        },
        market_data={
            "XXBTZUSD": [[ts, "100", "100", "100", "100", "100", "100", 1]],
        },
    )

    assert _materialize(postgres_engine).success

    with Session(postgres_engine) as session:
        rows = session.execute(select(ProviderAssetMarket)).scalars().all()
        assert len(rows) == 1
        assert rows[0].to_asset_id == kraken_base_data["btc_asset_id"]


@responses.activate
def test_filters_non_international_venue(
    postgres_engine: Engine, kraken_base_data: dict[str, Any]
) -> None:
    ts = int(dt.datetime.now().timestamp())
    _stub_kraken(
        asset_pairs={
            "XXBTZUSD": {
                "base": "XXBT",
                "quote": "ZUSD",
                "execution_venue": "international",
            },
            "XBTUSD:BTNL": {
                "base": "XXBT",
                "quote": "ZUSD",
                "execution_venue": "bitnomial_exchange",
            },
        },
        market_data={
            "XXBTZUSD": [[ts, "100", "100", "100", "100", "100", "100", 1]],
        },
    )

    assert _materialize(postgres_engine).success

    with Session(postgres_engine) as session:
        rows = session.execute(select(ProviderAssetMarket)).scalars().all()
        assert len(rows) == 1
        assert rows[0].close == 100.0


@pytest.mark.parametrize("batch_size", [1, 3, 100])
@responses.activate
def test_batches_large_dataset(
    monkeypatch: pytest.MonkeyPatch,
    postgres_engine: Engine,
    kraken_base_data: dict[str, Any],
    batch_size: int,
) -> None:
    monkeypatch.setattr(kraken_market_data, "BATCH_SIZE", batch_size)

    base_ts = int(dt.datetime.now().timestamp())
    n_per_pair = 5
    _stub_kraken(
        asset_pairs={
            "XXBTZUSD": {"base": "XXBT", "quote": "ZUSD"},
            "XETHZUSD": {"base": "XETH", "quote": "ZUSD"},
        },
        market_data={
            "XXBTZUSD": [
                [base_ts + 60 * i, "1", "1", "1", "1", "1", "1", 1]
                for i in range(n_per_pair)
            ],
            "XETHZUSD": [
                [base_ts + 60 * i, "2", "2", "2", "2", "2", "2", 1]
                for i in range(n_per_pair)
            ],
        },
    )

    assert _materialize(postgres_engine).success

    with Session(postgres_engine) as session:
        rows = session.execute(select(ProviderAssetMarket)).scalars().all()
        assert len(rows) == n_per_pair * 2


def _run_check_only(engine: Engine):
    return materialize(
        [kraken_provider_asset_market, kraken_market_data_quality],
        selection=AssetSelection.checks(kraken_market_data_quality),
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
    postgres_engine: Engine, kraken_base_data: dict[str, Any]
) -> None:
    # Whole-minute timestamp: round down to the minute.
    now = dt.datetime.now(dt.timezone.utc).replace(second=0, microsecond=0)
    ts = int(now.timestamp())
    _stub_kraken(
        asset_pairs={"XXBTZUSD": {"base": "XXBT", "quote": "ZUSD"}},
        market_data={
            "XXBTZUSD": [[ts, "100", "100", "100", "100", "100", "100", 1]],
        },
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
    postgres_engine: Engine, kraken_base_data: dict[str, Any]
) -> None:
    result = _run_check_only(postgres_engine)
    assert result.success
    evals = result.get_asset_check_evaluations()
    assert len(evals) == 1
    assert evals[0].passed is False
    assert "no rows" in (evals[0].description or "")


def test_data_quality_check_fails_on_off_minute_row(
    postgres_engine: Engine, kraken_base_data: dict[str, Any]
) -> None:
    base = dt.datetime.now(dt.timezone.utc).replace(
        second=0, microsecond=0, tzinfo=None
    )
    _seed_market_rows(
        postgres_engine,
        kraken_base_data,
        [
            base,
            base + dt.timedelta(seconds=37),  # off-minute
            base + dt.timedelta(minutes=2),
        ],
    )

    result = _run_check_only(postgres_engine)
    evals = result.get_asset_check_evaluations()
    assert evals[0].passed is False
    assert evals[0].metadata["off_minute_rows"].value == 1
    assert "whole-minute" in (evals[0].description or "")


def test_check_ignores_coinbase_rows(
    postgres_engine: Engine,
    kraken_base_data: dict[str, Any],
    coinbase_base_data: dict[str, Any],
) -> None:
    """Coinbase rows in the table must not affect the Kraken check."""
    base = dt.datetime.now(dt.timezone.utc).replace(
        second=0, microsecond=0, tzinfo=None
    )
    # Seed only Coinbase rows — the Kraken check should still report empty.
    with Session(postgres_engine) as session:
        session.add_all(
            [
                ProviderAssetMarket(
                    timestamp=base,
                    provider_id=coinbase_base_data["provider_id"],
                    from_asset_id=coinbase_base_data["usd_asset_id"],
                    to_asset_id=coinbase_base_data["btc_asset_id"],
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
