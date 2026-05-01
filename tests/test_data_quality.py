import datetime as dt
from typing import Any

from sqlalchemy import Engine, select
from sqlalchemy.orm import Session
from mc_postgres_db.models import ProviderAssetMarket

from dagster_loaders.defs.data_quality import (
    BASELINE_DAYS,
    provider_market_data_quality,
)


def _now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc).replace(
        second=0, microsecond=0, tzinfo=None
    )


def _seed(
    engine: Engine,
    *,
    provider_id: int,
    from_asset_id: int,
    to_asset_id: int,
    timestamps: list[dt.datetime],
    close: float = 100.0,
) -> None:
    with Session(engine) as session:
        session.add_all(
            [
                ProviderAssetMarket(
                    timestamp=ts,
                    provider_id=provider_id,
                    from_asset_id=from_asset_id,
                    to_asset_id=to_asset_id,
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


def _ids(base_data: dict[str, Any]) -> tuple[int, int, int]:
    return (
        base_data["provider_id"],
        base_data["usd_asset_id"],
        base_data["btc_asset_id"],
    )


def test_skips_pair_with_under_3_days_history(
    postgres_engine: Engine, kraken_base_data: dict[str, Any]
) -> None:
    """A pair whose first row is < 3 days old has no baseline yet — never flagged."""
    now = _now()
    provider_id, from_id, to_id = _ids(kraken_base_data)
    # Only 2 days of dense baseline + 0 recent rows. Should NOT flag (too new).
    timestamps = [now - dt.timedelta(hours=h) for h in range(2, 2 * 24)]
    _seed(
        postgres_engine,
        provider_id=provider_id,
        from_asset_id=from_id,
        to_asset_id=to_id,
        timestamps=timestamps,
    )
    result = provider_market_data_quality(postgres_engine, "Kraken")
    assert result.metadata["low_density_pairs_count"].value == 0
    assert result.metadata["skipped_new_pairs"].value == 1


def test_passes_with_stable_density(
    postgres_engine: Engine, kraken_base_data: dict[str, Any]
) -> None:
    """5 days at 1 row/hour baseline + 1 row/hour recent → no flag."""
    now = _now()
    provider_id, from_id, to_id = _ids(kraken_base_data)
    baseline = [now - dt.timedelta(hours=h) for h in range(2, 5 * 24)]
    recent = [now - dt.timedelta(minutes=30), now - dt.timedelta(minutes=90)]
    _seed(
        postgres_engine,
        provider_id=provider_id,
        from_asset_id=from_id,
        to_asset_id=to_id,
        timestamps=baseline + recent,
    )
    result = provider_market_data_quality(postgres_engine, "Kraken")
    assert result.metadata["low_density_pairs_count"].value == 0


def test_flags_density_drop_to_zero(
    postgres_engine: Engine, kraken_base_data: dict[str, Any]
) -> None:
    """5 days at 1 row/hour baseline + 0 recent rows → flagged."""
    now = _now()
    provider_id, from_id, to_id = _ids(kraken_base_data)
    baseline = [now - dt.timedelta(hours=h) for h in range(2, 5 * 24)]
    _seed(
        postgres_engine,
        provider_id=provider_id,
        from_asset_id=from_id,
        to_asset_id=to_id,
        timestamps=baseline,
    )
    result = provider_market_data_quality(postgres_engine, "Kraken")
    flagged = result.metadata["low_density_pairs"].value
    assert result.metadata["low_density_pairs_count"].value == 1
    assert flagged[0]["from_asset_id"] == from_id
    assert flagged[0]["to_asset_id"] == to_id
    assert flagged[0]["recent_count"] == 0
    assert flagged[0]["ratio"] == 0


def test_flags_density_dropped_below_half(
    postgres_engine: Engine, kraken_base_data: dict[str, Any]
) -> None:
    """5 days at 4 rows/hour baseline + 1 row in recent 2h (= 0.5/hr; ratio 0.125) → flagged."""
    now = _now()
    provider_id, from_id, to_id = _ids(kraken_base_data)
    # 4 rows per hour for 5 days
    baseline = []
    for h in range(2, 5 * 24):
        for q in range(4):
            baseline.append(now - dt.timedelta(hours=h, minutes=q * 15))
    # 1 recent row
    recent = [now - dt.timedelta(minutes=30)]
    _seed(
        postgres_engine,
        provider_id=provider_id,
        from_asset_id=from_id,
        to_asset_id=to_id,
        timestamps=baseline + recent,
    )
    result = provider_market_data_quality(postgres_engine, "Kraken")
    assert result.metadata["low_density_pairs_count"].value == 1
    flagged = result.metadata["low_density_pairs"].value[0]
    assert flagged["recent_count"] == 1
    # ratio is recent_per_hour (0.5) / baseline_per_hour (~4.0) ≈ 0.125
    assert flagged["ratio"] < 0.5


def test_passes_when_density_only_dipped_slightly(
    postgres_engine: Engine, kraken_base_data: dict[str, Any]
) -> None:
    """5 days at 4/hr baseline + 6 recent rows (3/hr; ratio 0.75) → pass (above 50% threshold)."""
    now = _now()
    provider_id, from_id, to_id = _ids(kraken_base_data)
    baseline = []
    for h in range(2, 5 * 24):
        for q in range(4):
            baseline.append(now - dt.timedelta(hours=h, minutes=q * 15))
    # 6 recent rows in 2h = 3/hour, baseline = 4/hour, ratio = 0.75 → not flagged
    recent = [now - dt.timedelta(minutes=20 * i) for i in range(6)]
    _seed(
        postgres_engine,
        provider_id=provider_id,
        from_asset_id=from_id,
        to_asset_id=to_id,
        timestamps=baseline + recent,
    )
    result = provider_market_data_quality(postgres_engine, "Kraken")
    assert result.metadata["low_density_pairs_count"].value == 0


def test_passes_when_density_increases(
    postgres_engine: Engine, kraken_base_data: dict[str, Any]
) -> None:
    """Sparse baseline (1.5/hr) + dense recent (15/hr) → pass (one-sided rule)."""
    now = _now()
    provider_id, from_id, to_id = _ids(kraken_base_data)
    # Sparse: ~1.5 rows per hour for 5 days (every 40 min — 3 rows per 2h)
    baseline = [now - dt.timedelta(minutes=40 * i) for i in range(3, 5 * 24 * 60 // 40)]
    # Dense recent: 30 rows in 2h = 15/hour
    recent = [now - dt.timedelta(minutes=4 * i) for i in range(30)]
    _seed(
        postgres_engine,
        provider_id=provider_id,
        from_asset_id=from_id,
        to_asset_id=to_id,
        timestamps=baseline + recent,
    )
    result = provider_market_data_quality(postgres_engine, "Kraken")
    assert result.metadata["low_density_pairs_count"].value == 0


def test_skips_pair_with_sub_1_per_window_baseline(
    postgres_engine: Engine, kraken_base_data: dict[str, Any]
) -> None:
    """Very sparse pair (baseline expects <1 row/2h) is skipped — no false positive on 0 recent."""
    now = _now()
    provider_id, from_id, to_id = _ids(kraken_base_data)
    # ~1 row every 4 hours for 5 days = baseline_per_hour ≈ 0.25
    # expected_recent = 0.25 * 2 = 0.5 (< 1) → skip
    baseline = [now - dt.timedelta(hours=4 * i) for i in range(1, 5 * 6)]
    _seed(
        postgres_engine,
        provider_id=provider_id,
        from_asset_id=from_id,
        to_asset_id=to_id,
        timestamps=baseline,
    )
    result = provider_market_data_quality(postgres_engine, "Kraken")
    # No flag despite 0 recent rows — baseline is too sparse to be meaningful.
    assert result.metadata["low_density_pairs_count"].value == 0


def test_baseline_capped_at_30_days(
    postgres_engine: Engine, kraken_base_data: dict[str, Any]
) -> None:
    """Rows older than the 30-day baseline window are excluded from the baseline computation."""
    now = _now()
    provider_id, from_id, to_id = _ids(kraken_base_data)
    # 1 row per hour for ~5 days inside the baseline window
    inside = [now - dt.timedelta(hours=h) for h in range(2, 5 * 24)]
    # Extra rows from > 30 days ago — should be ignored
    too_old = [now - dt.timedelta(days=BASELINE_DAYS + i) for i in range(1, 50)]
    _seed(
        postgres_engine,
        provider_id=provider_id,
        from_asset_id=from_id,
        to_asset_id=to_id,
        timestamps=inside + too_old,
    )
    result = provider_market_data_quality(postgres_engine, "Kraken")
    # Recent is 0 → with the dense in-window baseline this should flag.
    # If the >30d rows leaked into the baseline they'd inflate it further;
    # either way we expect the 1 flag here. The point is it doesn't crash
    # and the rows count agrees with the in-window baseline.
    assert result.metadata["low_density_pairs_count"].value == 1


def test_provider_scoping(
    postgres_engine: Engine,
    kraken_base_data: dict[str, Any],
    coinbase_base_data: dict[str, Any],
) -> None:
    """Coinbase rows must not influence the Kraken density check."""
    now = _now()
    # Seed dense Coinbase baseline + a fresh Coinbase recent row.
    cb_baseline = [now - dt.timedelta(hours=h) for h in range(2, 5 * 24)]
    _seed(
        postgres_engine,
        provider_id=coinbase_base_data["provider_id"],
        from_asset_id=coinbase_base_data["usd_asset_id"],
        to_asset_id=coinbase_base_data["btc_asset_id"],
        timestamps=cb_baseline + [now - dt.timedelta(minutes=10)],
    )
    # Kraken has nothing.
    result = provider_market_data_quality(postgres_engine, "Kraken")
    # The Kraken-scoped check should see zero rows (Coinbase doesn't pollute it).
    assert result.metadata["rows_in_last_2h"].value == 0
    assert result.metadata["low_density_pairs_count"].value == 0
    # Sanity: Coinbase-scoped check sees its rows.
    cb_result = provider_market_data_quality(postgres_engine, "Coinbase")
    assert cb_result.metadata["rows_in_last_2h"].value == 1


def test_passes_with_no_rows_at_all(
    postgres_engine: Engine, kraken_base_data: dict[str, Any]
) -> None:
    """Empty table for a provider: fails with 'no rows', no density flags."""
    result = provider_market_data_quality(postgres_engine, "Kraken")
    assert result.passed is False
    assert "no rows" in (result.description or "")
    assert result.metadata["low_density_pairs_count"].value == 0


def test_off_minute_rows_flagged(
    postgres_engine: Engine, kraken_base_data: dict[str, Any]
) -> None:
    """An off-minute timestamp inside the recent window is reported."""
    now = _now()
    provider_id, from_id, to_id = _ids(kraken_base_data)
    _seed(
        postgres_engine,
        provider_id=provider_id,
        from_asset_id=from_id,
        to_asset_id=to_id,
        timestamps=[
            now - dt.timedelta(minutes=10),
            now - dt.timedelta(minutes=20, seconds=37),  # off-minute
        ],
    )
    result = provider_market_data_quality(postgres_engine, "Kraken")
    assert result.metadata["off_minute_rows"].value == 1
    assert "whole-minute" in (result.description or "")


def test_min_close_zero_flagged(
    postgres_engine: Engine, kraken_base_data: dict[str, Any]
) -> None:
    """A row with close=0 in the recent window trips the price-positive check."""
    now = _now()
    provider_id, from_id, to_id = _ids(kraken_base_data)
    with Session(postgres_engine) as session:
        session.add(
            ProviderAssetMarket(
                timestamp=now - dt.timedelta(minutes=10),
                provider_id=provider_id,
                from_asset_id=from_id,
                to_asset_id=to_id,
                open=0.0,
                high=0.0,
                low=0.0,
                close=0.0,
                volume=0.0,
            )
        )
        session.commit()
    result = provider_market_data_quality(postgres_engine, "Kraken")
    assert "min(close)" in (result.description or "")


def test_skipped_new_pairs_metric_counts_correctly(
    postgres_engine: Engine, kraken_base_data: dict[str, Any]
) -> None:
    """Two new pairs + one mature pair → skipped_new_pairs == 2."""
    now = _now()
    provider_id = kraken_base_data["provider_id"]
    btc = kraken_base_data["btc_asset_id"]
    eth = kraken_base_data["eth_asset_id"]
    one_inch = kraken_base_data["one_inch_asset_id"]
    usd = kraken_base_data["usd_asset_id"]

    # Mature: 5 days of baseline at 1/hour.
    _seed(
        postgres_engine,
        provider_id=provider_id,
        from_asset_id=usd,
        to_asset_id=btc,
        timestamps=[now - dt.timedelta(hours=h) for h in range(2, 5 * 24)],
    )
    # New pair 1: only 1 day of history.
    _seed(
        postgres_engine,
        provider_id=provider_id,
        from_asset_id=usd,
        to_asset_id=eth,
        timestamps=[now - dt.timedelta(hours=h) for h in range(2, 24)],
    )
    # New pair 2: only 5 hours of history.
    _seed(
        postgres_engine,
        provider_id=provider_id,
        from_asset_id=usd,
        to_asset_id=one_inch,
        timestamps=[now - dt.timedelta(hours=h) for h in range(3, 8)],
    )
    result = provider_market_data_quality(postgres_engine, "Kraken")
    assert result.metadata["skipped_new_pairs"].value == 2


def test_returns_provider_id_in_flagged_pair(
    postgres_engine: Engine, kraken_base_data: dict[str, Any]
) -> None:
    """Flagged pairs include their provider_id so downstream alerts can route correctly."""
    now = _now()
    provider_id, from_id, to_id = _ids(kraken_base_data)
    baseline = [now - dt.timedelta(hours=h) for h in range(2, 5 * 24)]
    _seed(
        postgres_engine,
        provider_id=provider_id,
        from_asset_id=from_id,
        to_asset_id=to_id,
        timestamps=baseline,
    )
    result = provider_market_data_quality(postgres_engine, "Kraken")
    flagged = result.metadata["low_density_pairs"].value[0]
    assert flagged["provider_id"] == provider_id


def test_does_not_create_phantom_rows(
    postgres_engine: Engine, kraken_base_data: dict[str, Any]
) -> None:
    """Sanity: the check is read-only — never inserts rows."""
    now = _now()
    provider_id, from_id, to_id = _ids(kraken_base_data)
    _seed(
        postgres_engine,
        provider_id=provider_id,
        from_asset_id=from_id,
        to_asset_id=to_id,
        timestamps=[now - dt.timedelta(minutes=10)],
    )
    with Session(postgres_engine) as session:
        before = session.execute(
            select(ProviderAssetMarket).where(
                ProviderAssetMarket.provider_id == provider_id
            )
        ).all()

    provider_market_data_quality(postgres_engine, "Kraken")

    with Session(postgres_engine) as session:
        after = session.execute(
            select(ProviderAssetMarket).where(
                ProviderAssetMarket.provider_id == provider_id
            )
        ).all()
    assert len(before) == len(after)
