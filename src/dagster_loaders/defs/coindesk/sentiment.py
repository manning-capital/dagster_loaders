import datetime as dt

import nltk
import pandas as pd
from dagster import (
    AssetCheckResult,
    AssetCheckSeverity,
    AssetExecutionContext,
    Definitions,
    MaterializeResult,
    MetadataValue,
    asset,
    asset_check,
)
from mc_postgres_db.models import (
    ContentType,
    ProviderContent,
    ProviderContentSentiment,
    SentimentType,
)
from mc_postgres_db.operations import set_data
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sqlalchemy import func, or_, select
from sqlalchemy.orm import Session

from dagster_loaders.defs.coindesk.common import (
    COINDESK_SENTIMENT_POOL,
    FRESHNESS,
    RETRY,
)
from dagster_loaders.defs.coindesk.content import coindesk_news_content
from dagster_loaders.resources import PostgresResource


@asset(
    deps=[coindesk_news_content],
    pool=COINDESK_SENTIMENT_POOL,
    group_name="content",
    kinds={"python", "postgres", "ml"},
    owners=["glynfinck@gmail.com"],
    tags={"domain": "content", "kind": "sentiment"},
    retry_policy=RETRY,
    freshness_policy=FRESHNESS,
    description=(
        "NLTK VADER sentiment scores for today's NEWS content, written to "
        "`provider_content_sentiment`. Selects rows with NULL sentiment "
        "columns within today's UTC window via outer-join on "
        "`(provider_content_id, sentiment_type_id)`."
    ),
)
def coindesk_content_sentiment(
    context: AssetExecutionContext, postgres: PostgresResource
) -> MaterializeResult:
    engine = postgres.get_engine()
    try:
        nltk.download("vader_lexicon", quiet=True)

        with Session(engine) as session:
            news_content_type_id = session.execute(
                select(ContentType.id).where(ContentType.name == "NEWS")
            ).scalar_one()
            sentiment_type_id = session.execute(
                select(SentimentType.id).where(SentimentType.name == "NLTKVader")
            ).scalar_one()

        today = dt.date.today()
        start = dt.datetime.combine(today, dt.time.min)
        end = dt.datetime.combine(today, dt.time.max)

        unprocessed = pd.read_sql(
            select(ProviderContent.id, ProviderContent.content)
            .outerjoin(
                ProviderContentSentiment,
                (ProviderContent.id == ProviderContentSentiment.provider_content_id)
                & (ProviderContentSentiment.sentiment_type_id == sentiment_type_id),
            )
            .where(
                ProviderContent.content_type_id == news_content_type_id,
                ProviderContent.timestamp >= start,
                ProviderContent.timestamp <= end,
                or_(
                    ProviderContentSentiment.sentiment_score.is_(None),
                    ProviderContentSentiment.positive_sentiment_score.is_(None),
                    ProviderContentSentiment.negative_sentiment_score.is_(None),
                    ProviderContentSentiment.neutral_sentiment_score.is_(None),
                ),
            ),
            engine,
        )
        context.log.info(
            f"sentiment: {len(unprocessed)} unprocessed NEWS rows for "
            f"{today.isoformat()}"
        )

        if unprocessed.empty:
            return MaterializeResult(
                metadata={
                    "row_count": 0,
                    "table": ProviderContentSentiment.__tablename__,
                    "as_of": MetadataValue.text(today.isoformat()),
                }
            )

        analyzer = SentimentIntensityAnalyzer()
        raw_scores = unprocessed["content"].astype(str).apply(analyzer.polarity_scores)
        scored = pd.DataFrame(raw_scores.to_list()).rename(
            columns={
                "compound": "sentiment_score",
                "pos": "positive_sentiment_score",
                "neg": "negative_sentiment_score",
                "neu": "neutral_sentiment_score",
            }
        )
        scored["provider_content_id"] = unprocessed["id"].values
        scored["sentiment_type_id"] = sentiment_type_id

        set_data(engine, ProviderContentSentiment.__tablename__, scored, "upsert")

        return MaterializeResult(
            metadata={
                "row_count": len(scored),
                "mean_compound": float(scored["sentiment_score"].mean()),
                "table": ProviderContentSentiment.__tablename__,
                "as_of": MetadataValue.text(today.isoformat()),
                "preview": MetadataValue.md(
                    scored[
                        [
                            "provider_content_id",
                            "sentiment_score",
                            "positive_sentiment_score",
                            "negative_sentiment_score",
                            "neutral_sentiment_score",
                        ]
                    ]
                    .head()
                    .to_markdown(index=False)
                ),
            }
        )
    finally:
        engine.dispose()


@asset_check(
    asset=coindesk_content_sentiment,
    name="coindesk_content_sentiment_quality",
    description=(
        "Today's NLTK VADER sentiment rows have compound scores in [-1, 1], "
        "components in [0, 1], components summing to ~1.0, and every NEWS "
        "content row from today has a corresponding sentiment row."
    ),
    blocking=False,
)
def coindesk_content_sentiment_quality(
    postgres: PostgresResource,
) -> AssetCheckResult:
    engine = postgres.get_engine()
    try:
        with Session(engine) as session:
            news_content_type_id = session.execute(
                select(ContentType.id).where(ContentType.name == "NEWS")
            ).scalar_one()
            sentiment_type_id = session.execute(
                select(SentimentType.id).where(SentimentType.name == "NLTKVader")
            ).scalar_one()

        today = dt.date.today()
        start = dt.datetime.combine(today, dt.time.min)
        end = dt.datetime.combine(today, dt.time.max)

        score_rows = pd.read_sql(
            select(
                ProviderContentSentiment.sentiment_score,
                ProviderContentSentiment.positive_sentiment_score,
                ProviderContentSentiment.negative_sentiment_score,
                ProviderContentSentiment.neutral_sentiment_score,
            )
            .join(
                ProviderContent,
                ProviderContent.id == ProviderContentSentiment.provider_content_id,
            )
            .where(
                ProviderContentSentiment.sentiment_type_id == sentiment_type_id,
                ProviderContent.content_type_id == news_content_type_id,
                ProviderContent.timestamp >= start,
                ProviderContent.timestamp <= end,
            ),
            engine,
        )

        with Session(engine) as session:
            uncovered = session.execute(
                select(func.count())
                .select_from(ProviderContent)
                .outerjoin(
                    ProviderContentSentiment,
                    (ProviderContent.id == ProviderContentSentiment.provider_content_id)
                    & (ProviderContentSentiment.sentiment_type_id == sentiment_type_id),
                )
                .where(
                    ProviderContent.content_type_id == news_content_type_id,
                    ProviderContent.timestamp >= start,
                    ProviderContent.timestamp <= end,
                    ProviderContentSentiment.provider_content_id.is_(None),
                )
            ).scalar_one()
    finally:
        engine.dispose()

    out_of_range_compound = 0
    out_of_range_components = 0
    sum_off = 0
    if not score_rows.empty:
        compound = score_rows["sentiment_score"]
        out_of_range_compound = int(((compound < -1.0) | (compound > 1.0)).sum())
        for c in (
            "positive_sentiment_score",
            "negative_sentiment_score",
            "neutral_sentiment_score",
        ):
            col = score_rows[c]
            out_of_range_components += int(((col < 0.0) | (col > 1.0)).sum())
        component_sum = (
            score_rows["positive_sentiment_score"]
            + score_rows["negative_sentiment_score"]
            + score_rows["neutral_sentiment_score"]
        )
        sum_off = int(((component_sum - 1.0).abs() > 0.01).sum())

    failures: list[str] = []
    if out_of_range_compound:
        failures.append(f"{out_of_range_compound} compound scores outside [-1, 1]")
    if out_of_range_components:
        failures.append(f"{out_of_range_components} component scores outside [0, 1]")
    if sum_off:
        failures.append(f"{sum_off} rows where pos+neg+neu deviates from 1.0 by >0.01")
    if uncovered:
        failures.append(
            f"{uncovered} NEWS rows from today without a NLTKVader sentiment"
        )

    return AssetCheckResult(
        passed=not failures,
        severity=AssetCheckSeverity.WARN,
        description="; ".join(failures) if failures else "ok",
        metadata={
            "rows_today": len(score_rows),
            "out_of_range_compound": out_of_range_compound,
            "out_of_range_components": out_of_range_components,
            "sum_off_by_001": sum_off,
            "uncovered_news_today": int(uncovered),
            "as_of": MetadataValue.text(today.isoformat()),
        },
    )


defs = Definitions(
    assets=[coindesk_content_sentiment],
    asset_checks=[coindesk_content_sentiment_quality],
)
