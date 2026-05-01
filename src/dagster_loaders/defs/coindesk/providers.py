import pandas as pd
import requests
from dagster import (
    Definitions,
    MetadataValue,
    AssetCheckResult,
    MaterializeResult,
    AssetCheckSeverity,
    AssetExecutionContext,
    asset,
    asset_check,
)
from sqlalchemy import select
from sqlalchemy.orm import Session
from mc_postgres_db.models import Provider, ProviderType
from mc_postgres_db.operations import set_data

from dagster_loaders.utils import compare_dataframes
from dagster_loaders.resources import PostgresResource
from dagster_loaders.defs.coindesk.common import (
    RETRY,
    FRESHNESS,
    PROVIDER_COLUMNS,
    COINDESK_API_HOST,
    COINDESK_API_POOL,
)

_PROVIDER_TEXT_COLS = ("provider_external_code", "name", "url", "image_url")


def _coerce_provider_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    for col in _PROVIDER_TEXT_COLS:
        df[col] = df[col].astype(object)
    df["id"] = df["id"].astype("Int64")
    df["is_active"] = df["is_active"].astype("bool")
    df["provider_type_id"] = df["provider_type_id"].astype("Int64")
    df["underlying_provider_id"] = df["underlying_provider_id"].astype("Int64")
    return df[PROVIDER_COLUMNS]


@asset(
    pool=COINDESK_API_POOL,
    group_name="content",
    kinds={"python", "postgres"},
    owners=["glynfinck@gmail.com"],
    tags={"domain": "content", "provider": "coindesk"},
    retry_policy=RETRY,
    freshness_policy=FRESHNESS,
    description=(
        "Coindesk news source list upserted into `provider`. Pulls "
        "`/news/v1/source/list?lang=EN&status=ACTIVE`, normalizes columns, "
        "attaches existing ids by `provider_external_code`, and writes only "
        "added + changed rows via `set_data` upsert."
    ),
)
def coindesk_news_providers(
    context: AssetExecutionContext, postgres: PostgresResource
) -> MaterializeResult:
    engine = postgres.get_engine()
    try:
        with Session(engine) as session:
            coindesk_provider_id = session.execute(
                select(Provider.id).where(Provider.provider_external_code == "COINDESK")
            ).scalar_one()
            news_provider_type_id = session.execute(
                select(ProviderType.id).where(ProviderType.name == "NEWS_PROVIDER")
            ).scalar_one()

        existing = pd.read_sql(
            select(
                Provider.id,
                Provider.provider_external_code,
                Provider.name,
                Provider.url,
                Provider.image_url,
                Provider.is_active,
                Provider.provider_type_id,
                Provider.underlying_provider_id,
            ).where(
                Provider.underlying_provider_id == coindesk_provider_id,
                Provider.provider_type_id == news_provider_type_id,
            ),
            engine,
        )
        existing = _coerce_provider_dtypes(existing)

        resp = requests.get(
            f"{COINDESK_API_HOST}/news/v1/source/list",
            params={"lang": "EN", "status": "ACTIVE"},
            timeout=30,
        )
        resp.raise_for_status()
        raw = pd.DataFrame(resp.json()["Data"])
        context.log.info(f"Fetched {len(raw)} providers from Coindesk API")

        new = pd.DataFrame(
            {
                "provider_external_code": raw["ID"].astype(str),
                "name": raw["NAME"],
                "url": raw["URL"],
                "image_url": raw["IMAGE_URL"],
                "is_active": (raw["STATUS"] == "ACTIVE"),
                "provider_type_id": news_provider_type_id,
                "underlying_provider_id": coindesk_provider_id,
            }
        )
        new = new.merge(
            existing[["id", "provider_external_code"]].drop_duplicates(),
            on="provider_external_code",
            how="left",
        )
        new = _coerce_provider_dtypes(new)

        _, added, _, different = compare_dataframes(
            existing, new, ["provider_external_code"]
        )

        if not added.empty:
            set_data(
                engine,
                Provider.__tablename__,
                added.drop(columns=["id"]),
                "upsert",
            )
        if not different.empty:
            set_data(engine, Provider.__tablename__, different, "upsert")

        context.log.info(
            f"providers: {len(added)} added, {len(different)} updated, "
            f"{len(existing)} existing"
        )

        return MaterializeResult(
            metadata={
                "fetched": len(raw),
                "added": len(added),
                "updated": len(different),
                "existing": len(existing),
                "table": Provider.__tablename__,
                "preview": MetadataValue.md(
                    new.head().to_markdown(index=False) if not new.empty else "(empty)"
                ),
            }
        )
    finally:
        engine.dispose()


@asset_check(
    asset=coindesk_news_providers,
    name="coindesk_news_providers_quality",
    description=(
        "Coindesk news providers under the COINDESK parent are non-empty and "
        "every row has a non-null/non-empty external_code, name, and is_active flag."
    ),
    blocking=False,
)
def coindesk_news_providers_quality(postgres: PostgresResource) -> AssetCheckResult:
    engine = postgres.get_engine()
    try:
        with Session(engine) as session:
            coindesk_provider_id = session.execute(
                select(Provider.id).where(Provider.provider_external_code == "COINDESK")
            ).scalar_one()
            news_provider_type_id = session.execute(
                select(ProviderType.id).where(ProviderType.name == "NEWS_PROVIDER")
            ).scalar_one()

        rows = pd.read_sql(
            select(
                Provider.provider_external_code,
                Provider.name,
                Provider.is_active,
            ).where(
                Provider.underlying_provider_id == coindesk_provider_id,
                Provider.provider_type_id == news_provider_type_id,
            ),
            engine,
        )
    finally:
        engine.dispose()

    if rows.empty:
        return AssetCheckResult(
            passed=False,
            severity=AssetCheckSeverity.WARN,
            description="0 Coindesk news providers in DB",
            metadata={"total": 0},
        )

    null_code = int(
        (
            rows["provider_external_code"].isna()
            | (rows["provider_external_code"] == "")
        ).sum()
    )
    null_name = int((rows["name"].isna() | (rows["name"] == "")).sum())
    null_active = int(rows["is_active"].isna().sum())

    failures: list[str] = []
    if null_code > 0:
        failures.append(f"{null_code} rows with null/empty provider_external_code")
    if null_name > 0:
        failures.append(f"{null_name} rows with null/empty name")
    if null_active > 0:
        failures.append(f"{null_active} rows with null is_active")

    return AssetCheckResult(
        passed=not failures,
        severity=AssetCheckSeverity.WARN,
        description="; ".join(failures) if failures else "ok",
        metadata={
            "total": len(rows),
            "null_external_code": null_code,
            "null_name": null_name,
            "null_is_active": null_active,
        },
    )


defs = Definitions(
    assets=[coindesk_news_providers],
    asset_checks=[coindesk_news_providers_quality],
)
