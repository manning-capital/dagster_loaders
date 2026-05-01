import datetime as dt

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
from mc_postgres_db.models import Provider, ContentType, ProviderContent
from mc_postgres_db.operations import set_data

from dagster_loaders.utils import compare_dataframes
from dagster_loaders.resources import PostgresResource
from dagster_loaders.defs.coindesk.common import (
    RETRY,
    FRESHNESS,
    CONTENT_COLUMNS,
    CONTENT_LOOKBACK,
    COINDESK_API_HOST,
    COINDESK_API_POOL,
    RECENT_CONTENT_WINDOW,
)
from dagster_loaders.defs.coindesk.providers import coindesk_news_providers

_CONTENT_TEXT_COLS = ("content_external_code", "authors", "title", "content")


def _coerce_content_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    for col in _CONTENT_TEXT_COLS:
        df[col] = df[col].astype(object)
    df["id"] = df["id"].astype("Int64")
    df["timestamp"] = pd.to_datetime(df["timestamp"]).astype("datetime64[ns]")
    df["provider_id"] = df["provider_id"].astype("Int64")
    df["content_type_id"] = df["content_type_id"].astype("Int64")
    return df[CONTENT_COLUMNS]


@asset(
    deps=[coindesk_news_providers],
    pool=COINDESK_API_POOL,
    group_name="content",
    kinds={"python", "postgres"},
    owners=["glynfinck@gmail.com"],
    tags={"domain": "content", "provider": "coindesk"},
    retry_policy=RETRY,
    freshness_policy=FRESHNESS,
    description=(
        "Coindesk articles upserted into `provider_content`. Pulls "
        "`/news/v1/article/list` with a 2h lookback, maps `SOURCE_ID` to the "
        "provider id loaded by `coindesk_news_providers`, drops articles "
        "from unmapped sources, and writes only added + changed rows."
    ),
)
def coindesk_news_content(
    context: AssetExecutionContext, postgres: PostgresResource
) -> MaterializeResult:
    engine = postgres.get_engine()
    try:
        with Session(engine) as session:
            coindesk_provider_id = session.execute(
                select(Provider.id).where(Provider.provider_external_code == "COINDESK")
            ).scalar_one()
            news_content_type_id = session.execute(
                select(ContentType.id).where(ContentType.name == "NEWS")
            ).scalar_one()
            provider_rows = session.execute(
                select(Provider.id, Provider.provider_external_code).where(
                    Provider.underlying_provider_id == coindesk_provider_id
                )
            ).all()
        provider_map = {code: pid for pid, code in provider_rows}

        resp = requests.get(
            f"{COINDESK_API_HOST}/news/v1/article/list",
            params={
                "lang": "EN",
                "limit": 100,
                "to_ts": (dt.datetime.now() - CONTENT_LOOKBACK).timestamp(),
            },
            timeout=30,
        )
        resp.raise_for_status()
        raw = pd.DataFrame(resp.json()["Data"])
        context.log.info(f"Fetched {len(raw)} articles from Coindesk API")

        if raw.empty:
            return MaterializeResult(
                metadata={
                    "fetched": 0,
                    "added": 0,
                    "updated": 0,
                    "dropped_unmapped": 0,
                    "table": ProviderContent.__tablename__,
                }
            )

        new = pd.DataFrame(
            {
                "timestamp": raw["PUBLISHED_ON"].apply(dt.datetime.fromtimestamp),
                "provider_id": raw["SOURCE_ID"].astype(str).map(provider_map),
                "content_external_code": raw["ID"].astype(str),
                "content_type_id": news_content_type_id,
                "authors": raw["AUTHORS"],
                "title": raw["TITLE"],
                "content": raw["BODY"],
            }
        )

        unmapped_mask = new["provider_id"].isna()
        dropped_unmapped = int(unmapped_mask.sum())
        if dropped_unmapped:
            context.log.warning(
                f"Dropping {dropped_unmapped} articles whose SOURCE_ID is not in "
                f"the current Coindesk provider mapping"
            )
        new = new[~unmapped_mask].copy()

        empty_content_mask = new["content"].fillna("").astype(str).str.strip() == ""
        dropped_empty_content = int(empty_content_mask.sum())
        if dropped_empty_content:
            context.log.warning(
                f"Dropping {dropped_empty_content} articles with null/empty BODY"
            )
        new = new[~empty_content_mask].copy()

        if new.empty:
            return MaterializeResult(
                metadata={
                    "fetched": len(raw),
                    "added": 0,
                    "updated": 0,
                    "dropped_unmapped": dropped_unmapped,
                    "dropped_empty_content": dropped_empty_content,
                    "table": ProviderContent.__tablename__,
                }
            )

        ids = new["content_external_code"].drop_duplicates().tolist()
        existing = pd.read_sql(
            select(
                ProviderContent.id,
                ProviderContent.timestamp,
                ProviderContent.provider_id,
                ProviderContent.content_external_code,
                ProviderContent.content_type_id,
                ProviderContent.authors,
                ProviderContent.title,
                ProviderContent.content,
            ).where(ProviderContent.content_external_code.in_(ids)),
            engine,
        )

        existing = _coerce_content_dtypes(existing)

        new = new.merge(
            existing[["id", "content_external_code"]].drop_duplicates(),
            on="content_external_code",
            how="left",
        )
        new = _coerce_content_dtypes(new)

        _, added, _, different = compare_dataframes(
            existing, new, ["content_external_code"]
        )

        if not added.empty:
            set_data(
                engine,
                ProviderContent.__tablename__,
                added.drop(columns=["id"]),
                "upsert",
            )
        if not different.empty:
            set_data(engine, ProviderContent.__tablename__, different, "upsert")

        context.log.info(
            f"content: {len(added)} added, {len(different)} updated, "
            f"{dropped_unmapped} unmapped, {dropped_empty_content} empty-body"
        )

        return MaterializeResult(
            metadata={
                "fetched": len(raw),
                "added": len(added),
                "updated": len(different),
                "dropped_unmapped": dropped_unmapped,
                "dropped_empty_content": dropped_empty_content,
                "min_timestamp": MetadataValue.text(new["timestamp"].min().isoformat()),
                "max_timestamp": MetadataValue.text(new["timestamp"].max().isoformat()),
                "table": ProviderContent.__tablename__,
                "preview": MetadataValue.md(
                    new[["timestamp", "provider_id", "title"]]
                    .head()
                    .to_markdown(index=False)
                ),
            }
        )
    finally:
        engine.dispose()


@asset_check(
    asset=coindesk_news_content,
    name="coindesk_news_content_quality",
    description=(
        "Within the recent 4h window: NEWS content rows present, no null "
        "provider_id/title/content, and no rows with timestamps more than "
        "5min in the future."
    ),
    blocking=False,
)
def coindesk_news_content_quality(postgres: PostgresResource) -> AssetCheckResult:
    engine = postgres.get_engine()
    try:
        with Session(engine) as session:
            news_content_type_id = session.execute(
                select(ContentType.id).where(ContentType.name == "NEWS")
            ).scalar_one()

        cutoff = dt.datetime.now() - RECENT_CONTENT_WINDOW
        future_threshold = dt.datetime.now() + dt.timedelta(minutes=5)

        rows = pd.read_sql(
            select(
                ProviderContent.timestamp,
                ProviderContent.provider_id,
                ProviderContent.title,
                ProviderContent.content,
            ).where(
                ProviderContent.content_type_id == news_content_type_id,
                ProviderContent.timestamp >= cutoff,
            ),
            engine,
        )
    finally:
        engine.dispose()

    if rows.empty:
        return AssetCheckResult(
            passed=False,
            severity=AssetCheckSeverity.WARN,
            description="no NEWS content rows in last 4h",
            metadata={"rows_in_last_4h": 0},
        )

    null_provider = int(rows["provider_id"].isna().sum())
    null_title = int((rows["title"].isna() | (rows["title"] == "")).sum())
    null_content = int((rows["content"].isna() | (rows["content"] == "")).sum())
    timestamps = pd.to_datetime(rows["timestamp"])
    future_count = int((timestamps > future_threshold).sum())

    failures: list[str] = []
    if null_provider > 0:
        failures.append(f"{null_provider} rows with null provider_id")
    if null_title > 0:
        failures.append(f"{null_title} rows with null/empty title")
    if null_content > 0:
        failures.append(f"{null_content} rows with null/empty content")
    if future_count > 0:
        failures.append(f"{future_count} rows with timestamp >5min in the future")

    return AssetCheckResult(
        passed=not failures,
        severity=AssetCheckSeverity.WARN,
        description="; ".join(failures) if failures else "ok",
        metadata={
            "rows_in_last_4h": len(rows),
            "null_provider_id": null_provider,
            "null_title": null_title,
            "null_content": null_content,
            "future_timestamps": future_count,
            "max_timestamp": MetadataValue.text(timestamps.max().isoformat()),
        },
    )


defs = Definitions(
    assets=[coindesk_news_content],
    asset_checks=[coindesk_news_content_quality],
)
