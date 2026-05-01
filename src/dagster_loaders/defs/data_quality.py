import datetime as dt
import statistics
from typing import Any, Final, TypedDict

import pandas as pd
from dagster import MetadataValue, AssetCheckResult, AssetCheckSeverity
from sqlalchemy import Engine, case, func, select
from sqlalchemy.orm import Session, aliased
from mc_postgres_db.models import Asset, Provider, AssetType, ProviderAssetMarket

from dagster_loaders.utils import df_to_md_metadata

RECENT_WINDOW_HOURS: Final[int] = 2
BASELINE_DAYS: Final[int] = 30
MIN_HISTORY_DAYS: Final[int] = 3
DROP_RATIO: Final[float] = 0.5

CROSS_PROVIDER_WINDOW_MINUTES: Final[int] = 60
CROSS_PROVIDER_FFILL_BUFFER_MINUTES: Final[int] = 1440  # 24 hours
CROSS_PROVIDER_SPREAD_THRESHOLD: Final[float] = 0.10
FIAT_ASSET_TYPE_NAME: Final[str] = "FIAT_CURRENCY"


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


class CrossProviderFailingPair(TypedDict):
    from_asset_id: int
    to_asset_id: int
    providers: str
    median_spread: float
    max_spread: float
    n_minutes: int


def cross_provider_consistency_check(
    engine: Engine,
    spread_threshold: float = CROSS_PROVIDER_SPREAD_THRESHOLD,
    window_minutes: int = CROSS_PROVIDER_WINDOW_MINUTES,
    ffill_buffer_minutes: int = CROSS_PROVIDER_FFILL_BUFFER_MINUTES,
) -> AssetCheckResult:
    """Cross-provider close-price consistency guard.

    For every (from_asset, to_asset) pair, build a per-minute grid across the
    `window_minutes` ending at the most recent whole hour, forward-fill each
    provider's close onto the grid, then per minute compute
    `(max - min) / median` across providers (only minutes with >=2 providers
    count). Flag a pair if its **median** per-minute spread exceeds
    `spread_threshold`. Median across the minute grid is robust to a single
    misaligned minute — a real ticker-mapping bug stays high every minute, but
    a transient time-misalignment dilutes.

    Stable coins collapse to their fiat underlying (1 layer, fiat-only — so
    USDT/USDC -> USD but WBTC stays WBTC).

    `ffill_buffer_minutes` extends the SQL fetch back before `window_start` so
    the early minutes of the comparison window have a prior observation to
    forward-fill from. A provider with no rows in the buffer + window simply
    contributes no aligned data and the pair is skipped if fewer than 2
    providers remain. The buffer should comfortably exceed the loader cadence
    (30 min) so a single missed loader run doesn't kick a provider out.

    Designed to catch ticker-mapping bugs, not market microstructure noise —
    that's why the threshold is loose (10%).
    """
    now = dt.datetime.now(dt.timezone.utc).replace(tzinfo=None)
    window_end = now.replace(minute=0, second=0, microsecond=0)
    window_start = window_end - dt.timedelta(minutes=window_minutes)
    fetch_start = window_start - dt.timedelta(minutes=ffill_buffer_minutes)

    from_asset = aliased(Asset)
    to_asset = aliased(Asset)
    from_underlying = aliased(Asset)
    to_underlying = aliased(Asset)
    from_underlying_type = aliased(AssetType)
    to_underlying_type = aliased(AssetType)

    # Effective id: collapse to underlying iff underlying.asset_type is fiat.
    effective_from = case(
        (
            from_underlying_type.name == FIAT_ASSET_TYPE_NAME,
            from_asset.underlying_asset_id,
        ),
        else_=from_asset.id,
    ).label("effective_from_asset_id")
    effective_to = case(
        (
            to_underlying_type.name == FIAT_ASSET_TYPE_NAME,
            to_asset.underlying_asset_id,
        ),
        else_=to_asset.id,
    ).label("effective_to_asset_id")

    with Session(engine) as session:
        rows = session.execute(
            select(
                effective_from,
                effective_to,
                ProviderAssetMarket.provider_id,
                Provider.name.label("provider_name"),
                ProviderAssetMarket.timestamp,
                ProviderAssetMarket.close,
            )
            .join(Provider, Provider.id == ProviderAssetMarket.provider_id)
            .join(from_asset, from_asset.id == ProviderAssetMarket.from_asset_id)
            .join(to_asset, to_asset.id == ProviderAssetMarket.to_asset_id)
            .outerjoin(
                from_underlying, from_underlying.id == from_asset.underlying_asset_id
            )
            .outerjoin(
                from_underlying_type,
                from_underlying_type.id == from_underlying.asset_type_id,
            )
            .outerjoin(to_underlying, to_underlying.id == to_asset.underlying_asset_id)
            .outerjoin(
                to_underlying_type,
                to_underlying_type.id == to_underlying.asset_type_id,
            )
            .where(ProviderAssetMarket.timestamp >= fetch_start)
            .where(ProviderAssetMarket.timestamp <= window_end)
        ).all()

    if not rows:
        return AssetCheckResult(
            passed=False,
            severity=AssetCheckSeverity.WARN,
            description=(
                f"no rows in window {window_start.isoformat()} "
                f"to {window_end.isoformat()}"
            ),
            metadata={
                "pairs_evaluated": 0,
                "pairs_skipped_single_provider": 0,
                "failing_pairs_count": 0,
                "failing_pairs": df_to_md_metadata(
                    pd.DataFrame(), empty_placeholder="_No pairs evaluated._"
                ),
                "provider_stats": df_to_md_metadata(
                    pd.DataFrame(), empty_placeholder="_No rows fetched._"
                ),
                "window_minutes": window_minutes,
                "ffill_buffer_minutes": ffill_buffer_minutes,
                "window_start": MetadataValue.text(window_start.isoformat()),
                "window_end": MetadataValue.text(window_end.isoformat()),
                "threshold_pct": MetadataValue.float(spread_threshold * 100),
            },
        )

    df = pd.DataFrame(
        rows,
        columns=[
            "from_asset_id",
            "to_asset_id",
            "provider_id",
            "provider_name",
            "timestamp",
            "close",
        ],
    )
    df["close"] = df["close"].astype(float)

    # Resolve asset names for the effective ids (after stablecoin collapse) so
    # the metadata tables aren't just numeric ids.
    effective_asset_ids = sorted(
        set(df["from_asset_id"].tolist()) | set(df["to_asset_id"].tolist())
    )
    with Session(engine) as session:
        asset_name_rows = session.execute(
            select(Asset.id, Asset.name).where(Asset.id.in_(effective_asset_ids))
        ).all()
    asset_name_by_id: dict[int, str] = {int(aid): name for aid, name in asset_name_rows}

    # Per-provider stats — surface staleness/coverage at a glance regardless
    # of whether the check passes or fails. Show both id and name.
    provider_stats_df = (
        df.groupby(["provider_id", "provider_name"], as_index=False)
        .agg(
            rows=("close", "count"),
            pairs=(
                "from_asset_id",
                lambda s: (
                    df.loc[s.index, ["from_asset_id", "to_asset_id"]]
                    .drop_duplicates()
                    .shape[0]
                ),
            ),
            earliest=("timestamp", "min"),
            latest=("timestamp", "max"),
        )
        .sort_values("provider_name")
    )
    provider_stats_df["earliest"] = provider_stats_df["earliest"].astype(str)
    provider_stats_df["latest"] = provider_stats_df["latest"].astype(str)
    provider_stats_df = provider_stats_df[
        ["provider_id", "provider_name", "rows", "pairs", "earliest", "latest"]
    ]

    # Forward-fill each provider's close onto a per-minute grid covering
    # [window_start, window_end]. Rows in the buffer (before window_start) are
    # not in the grid but seed the ffill at minute 0.
    minute_grid = pd.date_range(start=window_start, end=window_end, freq="1min")
    aligned_records: list[dict[str, Any]] = []
    for (from_id, to_id, provider_id, provider_name), grp in df.groupby(
        ["from_asset_id", "to_asset_id", "provider_id", "provider_name"]
    ):
        series = grp.set_index("timestamp")["close"].sort_index()
        series = series[~series.index.duplicated(keep="last")]
        aligned = series.reindex(
            series.index.union(minute_grid).sort_values(), method=None
        ).ffill()
        aligned = aligned.reindex(minute_grid).dropna()
        for ts, close in aligned.items():
            aligned_records.append(
                {
                    "from_asset_id": int(from_id),
                    "to_asset_id": int(to_id),
                    "provider_id": int(provider_id),
                    "provider_name": provider_name,
                    "minute": ts,
                    "close": float(close),
                }
            )

    if not aligned_records:
        return AssetCheckResult(
            passed=False,
            severity=AssetCheckSeverity.WARN,
            description=(
                f"no rows in window {window_start.isoformat()} "
                f"to {window_end.isoformat()} after forward-fill"
            ),
            metadata={
                "pairs_evaluated": 0,
                "pairs_skipped_single_provider": 0,
                "failing_pairs_count": 0,
                "failing_pairs": df_to_md_metadata(
                    pd.DataFrame(), empty_placeholder="_No pairs evaluated._"
                ),
                "provider_stats": df_to_md_metadata(
                    provider_stats_df, empty_placeholder="_No rows fetched._"
                ),
                "window_minutes": window_minutes,
                "ffill_buffer_minutes": ffill_buffer_minutes,
                "window_start": MetadataValue.text(window_start.isoformat()),
                "window_end": MetadataValue.text(window_end.isoformat()),
                "threshold_pct": MetadataValue.float(spread_threshold * 100),
            },
        )

    aligned_df = pd.DataFrame(aligned_records)

    # Per-minute, per-pair spread across providers.
    per_minute = aligned_df.groupby(
        ["from_asset_id", "to_asset_id", "minute"], as_index=False
    ).agg(
        n_providers=("close", "count"),
        max_close=("close", "max"),
        min_close=("close", "min"),
        median_close=("close", "median"),
    )
    per_minute = per_minute[
        (per_minute["n_providers"] >= 2) & (per_minute["median_close"] > 0)
    ].copy()
    per_minute["spread"] = (
        per_minute["max_close"] - per_minute["min_close"]
    ) / per_minute["median_close"]

    # Per-pair: median spread across minutes (robust); also track max for
    # visibility into transient gaps.
    pair_agg = per_minute.groupby(["from_asset_id", "to_asset_id"], as_index=False).agg(
        median_spread=("spread", "median"),
        max_spread=("spread", "max"),
        n_minutes=("minute", "count"),
    )

    # Pairs that appeared in the data but never had >=2 providers at a single
    # minute -> skipped.
    pairs_in_data = aligned_df[["from_asset_id", "to_asset_id"]].drop_duplicates()
    pairs_skipped_single_provider = len(pairs_in_data) - len(pair_agg)
    pairs_evaluated = len(pair_agg)

    # Per-pair stats — analogous to provider_stats: every evaluated pair with
    # asset names + ids so coverage is scannable even when the check passes.
    pair_stats_df = pair_agg.copy()
    pair_stats_df["from_asset"] = pair_stats_df["from_asset_id"].map(
        lambda i: asset_name_by_id.get(int(i), str(i))
    )
    pair_stats_df["to_asset"] = pair_stats_df["to_asset_id"].map(
        lambda i: asset_name_by_id.get(int(i), str(i))
    )
    pair_stats_df["median_spread"] = pair_stats_df["median_spread"].round(4)
    pair_stats_df["max_spread"] = pair_stats_df["max_spread"].round(4)
    pair_stats_df = pair_stats_df.sort_values("median_spread", ascending=False)[
        [
            "from_asset",
            "from_asset_id",
            "to_asset",
            "to_asset_id",
            "n_minutes",
            "median_spread",
            "max_spread",
        ]
    ]

    failing_df = pair_agg[pair_agg["median_spread"] > spread_threshold].copy()
    failing_df = failing_df.sort_values("median_spread", ascending=False)

    # Enrich failing pairs with asset names + per-provider provider_name list
    # and the latest observed (timestamp, close) per provider so it's
    # diagnosable at a glance from the metadata table.
    if not failing_df.empty:
        latest_obs = (
            df.sort_values("timestamp")
            .drop_duplicates(
                subset=["from_asset_id", "to_asset_id", "provider_id"], keep="last"
            )
            .copy()
        )

        def _summarize(row: pd.Series) -> pd.Series:
            sub = latest_obs[
                (latest_obs["from_asset_id"] == row["from_asset_id"])
                & (latest_obs["to_asset_id"] == row["to_asset_id"])
            ].sort_values("provider_name")
            return pd.Series(
                {
                    "from_asset": asset_name_by_id.get(
                        int(row["from_asset_id"]), str(row["from_asset_id"])
                    ),
                    "to_asset": asset_name_by_id.get(
                        int(row["to_asset_id"]), str(row["to_asset_id"])
                    ),
                    "providers": ", ".join(sub["provider_name"].tolist()),
                    "latest_closes": ", ".join(
                        f"{c:.6g}" for c in sub["close"].tolist()
                    ),
                    "latest_timestamps": ", ".join(
                        t.isoformat() for t in sub["timestamp"].tolist()
                    ),
                }
            )

        summary = failing_df.apply(_summarize, axis=1)
        failing_df = pd.concat(
            [failing_df.reset_index(drop=True), summary.reset_index(drop=True)],
            axis=1,
        )
        failing_df = failing_df[
            [
                "from_asset",
                "to_asset",
                "from_asset_id",
                "to_asset_id",
                "providers",
                "latest_closes",
                "latest_timestamps",
                "median_spread",
                "max_spread",
                "n_minutes",
            ]
        ]
        failing_df["median_spread"] = failing_df["median_spread"].round(4)
        failing_df["max_spread"] = failing_df["max_spread"].round(4)

    median_spreads = pair_agg["median_spread"].tolist() if pairs_evaluated else []

    description = (
        "ok"
        if failing_df.empty
        else (
            f"{len(failing_df)} pair(s) median per-minute spread exceeded "
            f"{int(spread_threshold * 100)}%"
        )
    )

    return AssetCheckResult(
        passed=failing_df.empty,
        severity=AssetCheckSeverity.WARN,
        description=description,
        metadata={
            "pairs_evaluated": pairs_evaluated,
            "pairs_skipped_single_provider": pairs_skipped_single_provider,
            "failing_pairs_count": len(failing_df),
            "failing_pairs": df_to_md_metadata(
                failing_df, empty_placeholder="_No pairs above threshold._"
            ),
            "pair_stats": df_to_md_metadata(
                pair_stats_df,
                empty_placeholder="_No pairs evaluated._",
            ),
            "provider_stats": df_to_md_metadata(
                provider_stats_df,
                empty_placeholder="_No rows fetched._",
            ),
            "max_pair_median_spread": MetadataValue.float(
                round(max(median_spreads), 6) if median_spreads else 0.0
            ),
            "mean_pair_median_spread": MetadataValue.float(
                round(statistics.fmean(median_spreads), 6) if median_spreads else 0.0
            ),
            "median_pair_median_spread": MetadataValue.float(
                round(statistics.median(median_spreads), 6) if median_spreads else 0.0
            ),
            "window_minutes": window_minutes,
            "ffill_buffer_minutes": ffill_buffer_minutes,
            "window_start": MetadataValue.text(window_start.isoformat()),
            "window_end": MetadataValue.text(window_end.isoformat()),
            "threshold_pct": MetadataValue.float(spread_threshold * 100),
        },
    )
