import datetime as dt

from dagster import AssetSelection, materialize
from sqlalchemy import Engine, select
from sqlalchemy.orm import Session
from mc_postgres_db.models import (
    Provider,
    ProviderContent,
    ProviderContentSentiment,
)

from dagster_loaders.resources import PostgresResource
from dagster_loaders.defs.coindesk.sentiment import (
    coindesk_content_sentiment,
    coindesk_content_sentiment_quality,
)


def _seed_content_provider(engine: Engine, coindesk_base_data: dict[str, int]) -> int:
    with Session(engine) as session:
        provider = Provider(
            name="Source",
            description="Source",
            provider_external_code="42",
            is_active=True,
            provider_type_id=coindesk_base_data["news_provider_type_id"],
            underlying_provider_id=coindesk_base_data["coindesk_provider_id"],
        )
        session.add(provider)
        session.commit()
        return provider.id


def _seed_news_content(
    engine: Engine,
    coindesk_base_data: dict[str, int],
    *,
    provider_id: int,
    items: list[tuple[str, str]],  # (external_code, content)
    timestamp: dt.datetime | None = None,
) -> list[int]:
    if timestamp is None:
        # Mid-day today UTC so today filters always include it
        today = dt.date.today()
        timestamp = dt.datetime.combine(today, dt.time(12, 0))

    ids: list[int] = []
    with Session(engine) as session:
        for code, content in items:
            row = ProviderContent(
                timestamp=timestamp,
                provider_id=provider_id,
                content_external_code=code,
                content_type_id=coindesk_base_data["news_content_type_id"],
                authors="Editorial",
                title=f"Title for {code}",
                content=content,
            )
            session.add(row)
            session.commit()
            ids.append(row.id)
    return ids


def _materialize(engine: Engine):
    return materialize(
        [coindesk_content_sentiment],
        resources={
            "postgres": PostgresResource(
                url=engine.url.render_as_string(hide_password=False)
            )
        },
    )


def test_scores_today_content(
    postgres_engine: Engine, coindesk_base_data: dict[str, int]
) -> None:
    provider_id = _seed_content_provider(postgres_engine, coindesk_base_data)
    content_ids = _seed_news_content(
        postgres_engine,
        coindesk_base_data,
        provider_id=provider_id,
        items=[
            ("c1", "I love this market, everything is wonderful and great!"),
            ("c2", "This is terrible. Awful crash. Disaster."),
        ],
    )

    result = _materialize(postgres_engine)
    assert result.success

    with Session(postgres_engine) as session:
        rows = session.execute(select(ProviderContentSentiment)).scalars().all()
        assert len(rows) == 2
        by_content = {r.provider_content_id: r for r in rows}
        for cid in content_ids:
            assert cid in by_content
            r = by_content[cid]
            assert (
                r.sentiment_type_id
                == coindesk_base_data["nltk_vader_sentiment_type_id"]
            )
            assert -1.0 <= r.sentiment_score <= 1.0
            assert 0.0 <= r.positive_sentiment_score <= 1.0
            assert 0.0 <= r.negative_sentiment_score <= 1.0
            assert 0.0 <= r.neutral_sentiment_score <= 1.0
            total = (
                r.positive_sentiment_score
                + r.negative_sentiment_score
                + r.neutral_sentiment_score
            )
            assert abs(total - 1.0) < 0.01

        # Positive sample should score higher than negative one
        positive = by_content[content_ids[0]]
        negative = by_content[content_ids[1]]
        assert positive.sentiment_score > negative.sentiment_score


def test_skips_already_scored_content(
    postgres_engine: Engine, coindesk_base_data: dict[str, int]
) -> None:
    provider_id = _seed_content_provider(postgres_engine, coindesk_base_data)
    content_ids = _seed_news_content(
        postgres_engine,
        coindesk_base_data,
        provider_id=provider_id,
        items=[("c1", "Boring neutral text.")],
    )
    # Pre-populate sentiment row so the asset's filter excludes it
    with Session(postgres_engine) as session:
        session.add(
            ProviderContentSentiment(
                provider_content_id=content_ids[0],
                sentiment_type_id=coindesk_base_data["nltk_vader_sentiment_type_id"],
                sentiment_score=0.5,
                positive_sentiment_score=0.6,
                negative_sentiment_score=0.1,
                neutral_sentiment_score=0.3,
            )
        )
        session.commit()

    result = _materialize(postgres_engine)
    assert result.success

    with Session(postgres_engine) as session:
        rows = session.execute(select(ProviderContentSentiment)).scalars().all()
        assert len(rows) == 1
        # Original (pre-seeded) values should be preserved — asset did not re-score
        assert rows[0].sentiment_score == 0.5
        assert rows[0].positive_sentiment_score == 0.6


def test_no_op_when_no_content(
    postgres_engine: Engine, coindesk_base_data: dict[str, int]
) -> None:
    result = _materialize(postgres_engine)
    assert result.success

    with Session(postgres_engine) as session:
        rows = session.execute(select(ProviderContentSentiment)).scalars().all()
        assert rows == []


def test_skips_content_outside_today(
    postgres_engine: Engine, coindesk_base_data: dict[str, int]
) -> None:
    provider_id = _seed_content_provider(postgres_engine, coindesk_base_data)
    yesterday_noon = dt.datetime.combine(
        dt.date.today() - dt.timedelta(days=1), dt.time(12, 0)
    )
    _seed_news_content(
        postgres_engine,
        coindesk_base_data,
        provider_id=provider_id,
        items=[("c-old", "anything")],
        timestamp=yesterday_noon,
    )

    result = _materialize(postgres_engine)
    assert result.success

    with Session(postgres_engine) as session:
        rows = session.execute(select(ProviderContentSentiment)).scalars().all()
        assert rows == []


def _run_check_only(engine: Engine):
    return materialize(
        [coindesk_content_sentiment, coindesk_content_sentiment_quality],
        selection=AssetSelection.checks(coindesk_content_sentiment_quality),
        resources={
            "postgres": PostgresResource(
                url=engine.url.render_as_string(hide_password=False)
            )
        },
    )


def test_quality_check_passes_after_materialize(
    postgres_engine: Engine, coindesk_base_data: dict[str, int]
) -> None:
    provider_id = _seed_content_provider(postgres_engine, coindesk_base_data)
    _seed_news_content(
        postgres_engine,
        coindesk_base_data,
        provider_id=provider_id,
        items=[("c1", "Some news content for scoring.")],
    )
    assert _materialize(postgres_engine).success

    result = _run_check_only(postgres_engine)
    assert result.success
    evals = result.get_asset_check_evaluations()
    assert evals[0].passed is True
    assert evals[0].metadata["rows_today"].value == 1
    assert evals[0].metadata["uncovered_news_today"].value == 0


def test_quality_check_fails_on_uncovered_today_content(
    postgres_engine: Engine, coindesk_base_data: dict[str, int]
) -> None:
    provider_id = _seed_content_provider(postgres_engine, coindesk_base_data)
    _seed_news_content(
        postgres_engine,
        coindesk_base_data,
        provider_id=provider_id,
        items=[("uncovered", "content with no sentiment row")],
    )
    # Don't materialize — leaves uncovered

    result = _run_check_only(postgres_engine)
    evals = result.get_asset_check_evaluations()
    assert evals[0].passed is False
    assert "without a NLTKVader sentiment" in (evals[0].description or "")
    assert evals[0].metadata["uncovered_news_today"].value == 1


def test_skips_content_with_empty_body(
    postgres_engine: Engine, coindesk_base_data: dict[str, int]
) -> None:
    provider_id = _seed_content_provider(postgres_engine, coindesk_base_data)
    _seed_news_content(
        postgres_engine,
        coindesk_base_data,
        provider_id=provider_id,
        items=[("c-empty", ""), ("c-whitespace", "   \n  ")],
    )

    result = _materialize(postgres_engine)
    assert result.success

    with Session(postgres_engine) as session:
        rows = session.execute(select(ProviderContentSentiment)).scalars().all()
        assert rows == []


def test_quality_check_fails_on_legacy_all_zero_rows(
    postgres_engine: Engine, coindesk_base_data: dict[str, int]
) -> None:
    """All-zero sentiment rows (legacy data from empty-content scoring) should
    fail the check until they age out of the today window or get cleaned up
    manually. The asset itself no longer produces them."""
    provider_id = _seed_content_provider(postgres_engine, coindesk_base_data)
    content_ids = _seed_news_content(
        postgres_engine,
        coindesk_base_data,
        provider_id=provider_id,
        items=[("c1", "real content for scoring")],
    )
    with Session(postgres_engine) as session:
        session.add(
            ProviderContentSentiment(
                provider_content_id=content_ids[0],
                sentiment_type_id=coindesk_base_data["nltk_vader_sentiment_type_id"],
                sentiment_score=0.0,
                positive_sentiment_score=0.0,
                negative_sentiment_score=0.0,
                neutral_sentiment_score=0.0,
            )
        )
        session.commit()

    result = _run_check_only(postgres_engine)
    evals = result.get_asset_check_evaluations()
    assert evals[0].passed is False
    assert "all-zero rows" in (evals[0].description or "")
    assert evals[0].metadata["all_zero_rows"].value == 1
    # All-zero rows are reported separately and don't double-count toward sum_off
    assert evals[0].metadata["sum_off_by_001"].value == 0


def test_quality_check_fails_on_genuinely_off_sum(
    postgres_engine: Engine, coindesk_base_data: dict[str, int]
) -> None:
    provider_id = _seed_content_provider(postgres_engine, coindesk_base_data)
    content_ids = _seed_news_content(
        postgres_engine,
        coindesk_base_data,
        provider_id=provider_id,
        items=[("c1", "anything")],
    )
    with Session(postgres_engine) as session:
        # Components sum to 0.5 (not zero, not one) — genuine bug indicator
        session.add(
            ProviderContentSentiment(
                provider_content_id=content_ids[0],
                sentiment_type_id=coindesk_base_data["nltk_vader_sentiment_type_id"],
                sentiment_score=0.0,
                positive_sentiment_score=0.2,
                negative_sentiment_score=0.1,
                neutral_sentiment_score=0.2,
            )
        )
        session.commit()

    result = _run_check_only(postgres_engine)
    evals = result.get_asset_check_evaluations()
    assert evals[0].passed is False
    assert "deviates from 1.0" in (evals[0].description or "")
    assert evals[0].metadata["sum_off_by_001"].value == 1


def test_quality_check_fails_on_out_of_range_score(
    postgres_engine: Engine, coindesk_base_data: dict[str, int]
) -> None:
    provider_id = _seed_content_provider(postgres_engine, coindesk_base_data)
    content_ids = _seed_news_content(
        postgres_engine,
        coindesk_base_data,
        provider_id=provider_id,
        items=[("c1", "anything")],
    )
    with Session(postgres_engine) as session:
        session.add(
            ProviderContentSentiment(
                provider_content_id=content_ids[0],
                sentiment_type_id=coindesk_base_data["nltk_vader_sentiment_type_id"],
                sentiment_score=2.0,  # out of range
                positive_sentiment_score=0.5,
                negative_sentiment_score=0.2,
                neutral_sentiment_score=0.3,
            )
        )
        session.commit()

    result = _run_check_only(postgres_engine)
    evals = result.get_asset_check_evaluations()
    assert evals[0].passed is False
    assert "compound scores outside" in (evals[0].description or "")
    assert evals[0].metadata["out_of_range_compound"].value == 1
