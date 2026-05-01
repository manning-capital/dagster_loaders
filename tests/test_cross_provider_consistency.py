import datetime as dt
from typing import Any

from dagster import AssetCheckSeverity
from sqlalchemy import Engine, select
from sqlalchemy.orm import Session
from mc_postgres_db.models import (
    Asset,
    Provider,
    AssetType,
    ProviderType,
    ProviderAsset,
    ProviderAssetMarket,
)

from dagster_loaders.defs.data_quality import cross_provider_consistency_check


def _seed_market_row(
    session: Session,
    *,
    timestamp: dt.datetime,
    provider_id: int,
    from_asset_id: int,
    to_asset_id: int,
    close: float,
) -> None:
    session.add(
        ProviderAssetMarket(
            timestamp=timestamp,
            provider_id=provider_id,
            from_asset_id=from_asset_id,
            to_asset_id=to_asset_id,
            open=close,
            high=close,
            low=close,
            close=close,
            volume=1.0,
        )
    )


def _link_underlying(session: Session, asset_id: int, underlying_id: int) -> None:
    asset = session.execute(select(Asset).where(Asset.id == asset_id)).scalar_one()
    asset.underlying_asset_id = underlying_id
    session.commit()


def _window_end() -> dt.datetime:
    """The check snaps its window end to the most recent whole hour (naive UTC).

    Tests that want a row "inside the window" should seed at this anchor or
    within the prior `window_minutes` minutes; "outside the window" rows seed
    before that or after the anchor.
    """
    return dt.datetime.now(dt.timezone.utc).replace(
        minute=0, second=0, microsecond=0, tzinfo=None
    )


def test_passes_when_prices_agree_across_providers(
    postgres_engine: Engine,
    coinbase_base_data: dict[str, Any],
    kraken_base_data: dict[str, Any],
) -> None:
    """BTC-USD on both Coinbase and Kraken at the same price -> spread = 0, passes."""
    ts = _window_end()
    with Session(postgres_engine) as session:
        _seed_market_row(
            session,
            timestamp=ts,
            provider_id=coinbase_base_data["provider_id"],
            from_asset_id=coinbase_base_data["usd_asset_id"],
            to_asset_id=coinbase_base_data["btc_asset_id"],
            close=78000.0,
        )
        _seed_market_row(
            session,
            timestamp=ts,
            provider_id=kraken_base_data["provider_id"],
            from_asset_id=kraken_base_data["usd_asset_id"],
            to_asset_id=kraken_base_data["btc_asset_id"],
            close=78050.0,
        )
        session.commit()

    result = cross_provider_consistency_check(postgres_engine)
    assert result.passed is True
    assert result.severity == AssetCheckSeverity.WARN
    assert result.metadata["pairs_evaluated"].value == 1
    assert result.metadata["failing_pairs_count"].value == 0


def test_skips_pair_with_only_one_provider(
    postgres_engine: Engine, coinbase_base_data: dict[str, Any]
) -> None:
    ts = _window_end()
    with Session(postgres_engine) as session:
        _seed_market_row(
            session,
            timestamp=ts,
            provider_id=coinbase_base_data["provider_id"],
            from_asset_id=coinbase_base_data["usd_asset_id"],
            to_asset_id=coinbase_base_data["btc_asset_id"],
            close=78000.0,
        )
        session.commit()

    result = cross_provider_consistency_check(postgres_engine)
    assert result.passed is True
    assert result.metadata["pairs_evaluated"].value == 0
    assert result.metadata["pairs_skipped_single_provider"].value == 1


def test_fails_on_15_percent_spread(
    postgres_engine: Engine,
    coinbase_base_data: dict[str, Any],
    kraken_base_data: dict[str, Any],
) -> None:
    """Coinbase BTC at $78k, Kraken BTC at $90k -> ~14% spread; fails."""
    ts = _window_end()
    with Session(postgres_engine) as session:
        _seed_market_row(
            session,
            timestamp=ts,
            provider_id=coinbase_base_data["provider_id"],
            from_asset_id=coinbase_base_data["usd_asset_id"],
            to_asset_id=coinbase_base_data["btc_asset_id"],
            close=78000.0,
        )
        _seed_market_row(
            session,
            timestamp=ts,
            provider_id=kraken_base_data["provider_id"],
            from_asset_id=kraken_base_data["usd_asset_id"],
            to_asset_id=kraken_base_data["btc_asset_id"],
            close=90000.0,
        )
        session.commit()

    result = cross_provider_consistency_check(postgres_engine)
    assert result.passed is False
    assert result.metadata["failing_pairs_count"].value == 1
    assert "exceeded 10% spread" in (result.description or "")


def test_usdt_collapses_to_usd_via_underlying(
    postgres_engine: Engine,
    coinbase_base_data: dict[str, Any],
    okx_base_data: dict[str, Any],
) -> None:
    """Coinbase BTC-USD at $78k and OKX BTC-USDT at $78k must compare as the
    same (from, to) pair once USDT collapses to USD via underlying_asset_id.
    """
    ts = _window_end()
    with Session(postgres_engine) as session:
        _link_underlying(
            session,
            okx_base_data["usdt_asset_id"],
            coinbase_base_data["usd_asset_id"],
        )

        _seed_market_row(
            session,
            timestamp=ts,
            provider_id=coinbase_base_data["provider_id"],
            from_asset_id=coinbase_base_data["usd_asset_id"],
            to_asset_id=coinbase_base_data["btc_asset_id"],
            close=78000.0,
        )
        _seed_market_row(
            session,
            timestamp=ts,
            provider_id=okx_base_data["provider_id"],
            from_asset_id=okx_base_data["usdt_asset_id"],
            to_asset_id=okx_base_data["btc_asset_id"],
            close=78050.0,
        )
        session.commit()

    result = cross_provider_consistency_check(postgres_engine)
    assert result.passed is True
    assert result.metadata["pairs_evaluated"].value == 1
    assert result.metadata["pairs_skipped_single_provider"].value == 0


def test_wrapped_assets_do_not_collapse_to_underlying(
    postgres_engine: Engine,
    coinbase_base_data: dict[str, Any],
    kraken_base_data: dict[str, Any],
) -> None:
    """WBTC has BTC as underlying, but BTC is CryptoCurrency (not FiatCurrency)
    so collapse must NOT happen. Coinbase WBTC-USD and Kraken BTC-USD stay as
    two separate (to_asset) groups, each with one provider -> both skipped.
    """
    ts = _window_end()
    with Session(postgres_engine) as session:
        crypto_type = session.execute(
            select(AssetType).where(AssetType.name == "CryptoCurrency")
        ).scalar_one()
        wbtc = Asset(
            name="WBTC",
            description="WBTC",
            asset_type_id=crypto_type.id,
            underlying_asset_id=coinbase_base_data["btc_asset_id"],
        )
        session.add(wbtc)
        session.commit()
        wbtc_id = wbtc.id
        session.add(
            ProviderAsset(
                date=(dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=1)).date(),
                provider_id=coinbase_base_data["provider_id"],
                asset_id=wbtc_id,
                asset_code="WBTC",
                is_active=True,
            )
        )
        session.commit()

        _seed_market_row(
            session,
            timestamp=ts,
            provider_id=coinbase_base_data["provider_id"],
            from_asset_id=coinbase_base_data["usd_asset_id"],
            to_asset_id=wbtc_id,
            close=78000.0,
        )
        _seed_market_row(
            session,
            timestamp=ts,
            provider_id=kraken_base_data["provider_id"],
            from_asset_id=kraken_base_data["usd_asset_id"],
            to_asset_id=kraken_base_data["btc_asset_id"],
            close=78000.0,
        )
        session.commit()

    result = cross_provider_consistency_check(postgres_engine)
    assert result.metadata["pairs_evaluated"].value == 0
    assert result.metadata["pairs_skipped_single_provider"].value == 2


def test_wrong_mapping_detection(
    postgres_engine: Engine,
    coinbase_base_data: dict[str, Any],
    kraken_base_data: dict[str, Any],
) -> None:
    """Simulates the bug this check exists to catch: Coinbase has a row tagged
    to_asset = ETH but with a BTC-priced close, while Kraken ETH-USD has the
    real ETH price. Spread is enormous, check fails, ETH pair shows up in
    failing_pairs.
    """
    ts = _window_end()
    with Session(postgres_engine) as session:
        _seed_market_row(
            session,
            timestamp=ts,
            provider_id=coinbase_base_data["provider_id"],
            from_asset_id=coinbase_base_data["usd_asset_id"],
            to_asset_id=coinbase_base_data["eth_asset_id"],
            close=78000.0,
        )
        _seed_market_row(
            session,
            timestamp=ts,
            provider_id=kraken_base_data["provider_id"],
            from_asset_id=kraken_base_data["usd_asset_id"],
            to_asset_id=kraken_base_data["eth_asset_id"],
            close=2200.0,
        )
        session.commit()

    result = cross_provider_consistency_check(postgres_engine)
    assert result.passed is False
    assert result.metadata["failing_pairs_count"].value == 1


def test_window_cutoff_excludes_stale_rows(
    postgres_engine: Engine,
    coinbase_base_data: dict[str, Any],
    kraken_base_data: dict[str, Any],
) -> None:
    """Window is `[anchor - window_minutes, anchor]` where anchor =
    floor(now, hour). Coinbase row sits just before the window start (excluded);
    Kraken row is inside. Only one provider in window -> pair is skipped.

    Tests against the configurable window param so the assertion stays valid
    if the default is tuned later.
    """
    window_minutes = 30
    anchor = _window_end()
    with Session(postgres_engine) as session:
        _seed_market_row(
            session,
            timestamp=anchor - dt.timedelta(minutes=window_minutes + 1),
            provider_id=coinbase_base_data["provider_id"],
            from_asset_id=coinbase_base_data["usd_asset_id"],
            to_asset_id=coinbase_base_data["btc_asset_id"],
            close=78000.0,
        )
        _seed_market_row(
            session,
            timestamp=anchor - dt.timedelta(minutes=1),
            provider_id=kraken_base_data["provider_id"],
            from_asset_id=kraken_base_data["usd_asset_id"],
            to_asset_id=kraken_base_data["btc_asset_id"],
            close=78000.0,
        )
        session.commit()

    result = cross_provider_consistency_check(
        postgres_engine, window_minutes=window_minutes
    )
    assert result.metadata["pairs_evaluated"].value == 0
    assert result.metadata["pairs_skipped_single_provider"].value == 1


def test_post_anchor_rows_excluded(
    postgres_engine: Engine,
    coinbase_base_data: dict[str, Any],
    kraken_base_data: dict[str, Any],
) -> None:
    """Rows whose timestamp is *after* the snapped window end (i.e. between
    the most recent whole hour and now) are excluded. This is the whole point
    of snapping: each run reads a deterministic clock-aligned bucket, not a
    rolling slice that includes the in-progress hour.
    """
    anchor = _window_end()
    with Session(postgres_engine) as session:
        _seed_market_row(
            session,
            timestamp=anchor + dt.timedelta(minutes=5),
            provider_id=coinbase_base_data["provider_id"],
            from_asset_id=coinbase_base_data["usd_asset_id"],
            to_asset_id=coinbase_base_data["btc_asset_id"],
            close=78000.0,
        )
        _seed_market_row(
            session,
            timestamp=anchor + dt.timedelta(minutes=10),
            provider_id=kraken_base_data["provider_id"],
            from_asset_id=kraken_base_data["usd_asset_id"],
            to_asset_id=kraken_base_data["btc_asset_id"],
            close=99999.0,
        )
        session.commit()

    result = cross_provider_consistency_check(postgres_engine)
    assert result.passed is False
    assert "no rows in window" in (result.description or "")


def test_empty_table_marks_check_as_failed(postgres_engine: Engine) -> None:
    """No data in the recent window must surface as a failure, not a silent
    pass — otherwise an outage in all loaders would look healthy."""
    with Session(postgres_engine) as session:
        provider_type = ProviderType(name="ProviderType", description="x")
        session.add(provider_type)
        session.commit()
        provider = Provider(
            name="DummyProvider", description="x", provider_type_id=provider_type.id
        )
        session.add(provider)
        session.commit()

    result = cross_provider_consistency_check(postgres_engine)
    assert result.passed is False
    assert "no rows" in (result.description or "")


def test_threshold_configurable(
    postgres_engine: Engine,
    coinbase_base_data: dict[str, Any],
    kraken_base_data: dict[str, Any],
) -> None:
    """A 5% real spread passes at default 10% threshold but fails at 1%."""
    ts = _window_end()
    with Session(postgres_engine) as session:
        _seed_market_row(
            session,
            timestamp=ts,
            provider_id=coinbase_base_data["provider_id"],
            from_asset_id=coinbase_base_data["usd_asset_id"],
            to_asset_id=coinbase_base_data["btc_asset_id"],
            close=78000.0,
        )
        _seed_market_row(
            session,
            timestamp=ts,
            provider_id=kraken_base_data["provider_id"],
            from_asset_id=kraken_base_data["usd_asset_id"],
            to_asset_id=kraken_base_data["btc_asset_id"],
            close=82000.0,
        )
        session.commit()

    assert cross_provider_consistency_check(postgres_engine).passed is True
    assert (
        cross_provider_consistency_check(postgres_engine, spread_threshold=0.01).passed
        is False
    )
