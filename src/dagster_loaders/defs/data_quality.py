import datetime as dt
import statistics
from typing import Final, TypedDict

import pandas as pd
from dagster import MetadataValue, AssetCheckResult, AssetCheckSeverity
from sqlalchemy import Engine, func, select
from sqlalchemy.orm import Session
from mc_postgres_db.models import Provider, ProviderAssetMarket

from dagster_loaders.utils import df_to_md_metadata

RECENT_WINDOW_HOURS: Final[int] = 2
BASELINE_DAYS: Final[int] = 30
MIN_HISTORY_DAYS: Final[int] = 3
DROP_RATIO: Final[float] = 0.5


class LowDensityPair(TypedDict):
    from_asset_id: int
    to_asset_id: int
    provider_id: int
    recent_count: int
    recent_per_hour: float
    baseline_per_hour: float
    ratio: float


def _low_density_pairs_md(pairs: list[LowDensityPair]) -> MetadataValue:
    sorted_pairs = sorted(pairs, key=lambda p: p["ratio"])
    df = pd.DataFrame(sorted_pairs)
    return df_to_md_metadata(df, empty_placeholder="_No pairs below threshold._")


def provider_market_data_quality(
    engine: Engine, provider_name: str
) -> AssetCheckResult:
    """Standard provider market-data quality check.

    Checks: rows present in the last 2h, all close prices > 0, every timestamp
    aligned to a whole-minute boundary, and no per-pair density drop below 50%
    of the pair's own 30-day baseline. Pairs whose first row is < 3 days old
    are skipped (no baseline yet). The density rule is one-sided: density
    rising above baseline never trips.
    """
    with Session(engine) as session:
        provider_id = session.execute(
            select(Provider.id).where(Provider.name == provider_name)
        ).scalar_one()

        now = dt.datetime.now(dt.timezone.utc).replace(tzinfo=None)
        recent_cutoff = now - dt.timedelta(hours=RECENT_WINDOW_HOURS)
        historical_cutoff = now - dt.timedelta(days=BASELINE_DAYS)

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

        recent_stats = session.execute(
            select(
                ProviderAssetMarket.from_asset_id,
                ProviderAssetMarket.to_asset_id,
                func.count().label("recent_count"),
            )
            .where(
                ProviderAssetMarket.provider_id == provider_id,
                ProviderAssetMarket.timestamp >= recent_cutoff,
            )
            .group_by(
                ProviderAssetMarket.from_asset_id,
                ProviderAssetMarket.to_asset_id,
            )
        ).all()

        historical_stats = session.execute(
            select(
                ProviderAssetMarket.from_asset_id,
                ProviderAssetMarket.to_asset_id,
                func.count().label("historical_count"),
                func.min(ProviderAssetMarket.timestamp).label("earliest_ts"),
            )
            .where(
                ProviderAssetMarket.provider_id == provider_id,
                ProviderAssetMarket.timestamp >= historical_cutoff,
                ProviderAssetMarket.timestamp < recent_cutoff,
            )
            .group_by(
                ProviderAssetMarket.from_asset_id,
                ProviderAssetMarket.to_asset_id,
            )
        ).all()

    recent_by_pair = {
        (r.from_asset_id, r.to_asset_id): r.recent_count for r in recent_stats
    }
    historical_by_pair = {(h.from_asset_id, h.to_asset_id): h for h in historical_stats}

    low_density_pairs: list[LowDensityPair] = []
    pair_ratios: list[float] = []
    skipped_new_pairs: int = 0
    skipped_sparse_pairs: int = 0

    historical_window_hours_max: int = BASELINE_DAYS * 24 - RECENT_WINDOW_HOURS

    for key, hist in historical_by_pair.items():
        from_asset_id, to_asset_id = key
        history_age_hours = (now - hist.earliest_ts).total_seconds() / 3600
        if history_age_hours < MIN_HISTORY_DAYS * 24:
            skipped_new_pairs += 1
            continue

        historical_hours = min(
            history_age_hours - RECENT_WINDOW_HOURS,
            historical_window_hours_max,
        )
        if historical_hours <= 0:
            skipped_new_pairs += 1
            continue

        baseline_per_hour = hist.historical_count / historical_hours
        expected_recent = baseline_per_hour * RECENT_WINDOW_HOURS

        # Skip pairs whose baseline expects < 1 row in the recent window;
        # the ratio test isn't statistically meaningful at that density and
        # would routinely false-positive on naturally sparse pairs.
        if expected_recent < 1:
            skipped_sparse_pairs += 1
            continue

        recent_count = recent_by_pair.get(key, 0)
        recent_per_hour = recent_count / RECENT_WINDOW_HOURS
        ratio = recent_per_hour / baseline_per_hour
        pair_ratios.append(ratio)

        if recent_per_hour < baseline_per_hour * DROP_RATIO:
            low_density_pairs.append(
                LowDensityPair(
                    from_asset_id=from_asset_id,
                    to_asset_id=to_asset_id,
                    provider_id=provider_id,
                    recent_count=recent_count,
                    recent_per_hour=round(recent_per_hour, 3),
                    baseline_per_hour=round(baseline_per_hour, 3),
                    ratio=round(ratio, 3),
                )
            )

    pairs_evaluated: int = len(pair_ratios)
    mean_ratio: float = statistics.fmean(pair_ratios) if pair_ratios else 0.0
    median_ratio: float = statistics.median(pair_ratios) if pair_ratios else 0.0
    min_ratio: float = min(pair_ratios) if pair_ratios else 0.0
    max_ratio: float = max(pair_ratios) if pair_ratios else 0.0
    pairs_below_75pct: int = sum(1 for r in pair_ratios if r < 0.75)
    pairs_below_50pct: int = sum(1 for r in pair_ratios if r < 0.50)
    pairs_below_25pct: int = sum(1 for r in pair_ratios if r < 0.25)

    failures: list[str] = []
    if recent_rows == 0:
        failures.append("no rows with timestamp in the last 2h")
    if min_close is not None and min_close <= 0:
        failures.append(f"min(close) = {min_close} (expected > 0)")
    if off_minute_rows > 0:
        failures.append(
            f"{off_minute_rows} rows are not aligned to a whole-minute boundary"
        )
    if low_density_pairs:
        failures.append(
            f"{len(low_density_pairs)} pair(s) dropped below "
            f"{int(DROP_RATIO * 100)}% of their {BASELINE_DAYS}d baseline density"
        )

    return AssetCheckResult(
        passed=not failures,
        severity=AssetCheckSeverity.WARN,
        description="; ".join(failures) if failures else "ok",
        metadata={
            "rows_in_last_2h": recent_rows,
            "min_close_price": MetadataValue.float(float(min_close or 0.0)),
            "off_minute_rows": off_minute_rows,
            "low_density_pairs_count": len(low_density_pairs),
            "low_density_pairs": _low_density_pairs_md(low_density_pairs),
            "skipped_new_pairs": skipped_new_pairs,
            "skipped_sparse_pairs": skipped_sparse_pairs,
            "pairs_evaluated": pairs_evaluated,
            "mean_density_ratio": MetadataValue.float(round(mean_ratio, 4)),
            "median_density_ratio": MetadataValue.float(round(median_ratio, 4)),
            "min_density_ratio": MetadataValue.float(round(min_ratio, 4)),
            "max_density_ratio": MetadataValue.float(round(max_ratio, 4)),
            "pairs_below_75pct": pairs_below_75pct,
            "pairs_below_50pct": pairs_below_50pct,
            "pairs_below_25pct": pairs_below_25pct,
        },
    )
