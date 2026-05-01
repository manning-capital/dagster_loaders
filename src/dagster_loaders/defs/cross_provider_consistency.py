from typing import Final
from collections.abc import Iterable

from dagster import (
    Definitions,
    AssetCheckKey,
    AssetCheckSpec,
    AssetSelection,
    AssetCheckResult,
    ScheduleDefinition,
    AssetCheckExecutionContext,
    define_asset_job,
    multi_asset_check,
)

from dagster_loaders.resources import PostgresResource
from dagster_loaders.defs.data_quality import cross_provider_consistency_check
from dagster_loaders.defs.okx.market_data import okx_provider_asset_market
from dagster_loaders.defs.kraken.market_data import kraken_provider_asset_market
from dagster_loaders.defs.coinbase.market_data import coinbase_provider_asset_market

CHECK_NAME: Final[str] = "cross_provider_consistency"

UPSTREAM_ASSETS = (
    coinbase_provider_asset_market,
    kraken_provider_asset_market,
    okx_provider_asset_market,
)

CHECK_DESCRIPTION: Final[str] = (
    "Cross-provider close-price spread per (from, to) asset pair within the "
    "last 30 minutes. Stable coins are collapsed to their fiat underlying "
    "(USDT/USDC -> USD) but wrapped tokens (WBTC, cbBTC) are not. Pairs with "
    "<2 providers are skipped. Fails if any pair's (max - min) / median close "
    "exceeds 10% — typically indicates a ticker-mapping bug, not a price move."
)


@multi_asset_check(
    specs=[
        AssetCheckSpec(
            name=CHECK_NAME,
            asset=a.key,
            additional_deps=[other.key for other in UPSTREAM_ASSETS if other is not a],
            description=CHECK_DESCRIPTION,
            blocking=False,
        )
        for a in UPSTREAM_ASSETS
    ],
    can_subset=True,
)
def cross_provider_consistency(
    context: AssetCheckExecutionContext,
    postgres: PostgresResource,
) -> Iterable[AssetCheckResult]:
    # Subsetting is required because each per-provider job (e.g. coinbase_market_job)
    # selects only one of these specs' check keys; without can_subset, Dagster fails
    # to build those jobs at definition load time.
    selected = context.selected_asset_check_keys
    engine = postgres.get_engine()
    try:
        result = cross_provider_consistency_check(engine)
    finally:
        engine.dispose()
    for a in UPSTREAM_ASSETS:
        key = AssetCheckKey(asset_key=a.key, name=CHECK_NAME)
        if key not in selected:
            continue
        yield AssetCheckResult(
            check_name=CHECK_NAME,
            asset_key=a.key,
            passed=result.passed,
            severity=result.severity,
            description=result.description,
            metadata=result.metadata,
        )


# Unified pipeline: all three loaders + their per-provider DQ checks + the
# cross-provider check, on one 30-min schedule. Dagster orders the assets
# first, then the cross-provider check (via additional_deps), so the check
# always runs against the freshest data. The check is a multi_asset_check
# yielding one result per upstream asset, each routed to that asset's UI page.
market_data_job = define_asset_job(
    name="market_data_job",
    selection=AssetSelection.assets(
        coinbase_provider_asset_market,
        kraken_provider_asset_market,
        okx_provider_asset_market,
    ),
)

market_data_schedule = ScheduleDefinition(
    name="market_data_every_30min",
    cron_schedule="*/30 * * * *",
    job=market_data_job,
)


defs = Definitions(
    asset_checks=[cross_provider_consistency],
    jobs=[market_data_job],
    schedules=[market_data_schedule],
)
