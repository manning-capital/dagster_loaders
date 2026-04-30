import datetime as dt
from typing import Any

import responses
from dagster import AssetSelection, materialize
from mc_postgres_db.models import Provider, ProviderContent
from sqlalchemy import Engine, select
from sqlalchemy.orm import Session

from dagster_loaders.defs.coindesk.common import COINDESK_API_HOST
from dagster_loaders.defs.coindesk.content import (
    coindesk_news_content,
    coindesk_news_content_quality,
)
from dagster_loaders.defs.coindesk.providers import coindesk_news_providers
from dagster_loaders.resources import PostgresResource


ARTICLE_LIST_URL = f"{COINDESK_API_HOST}/news/v1/article/list"


def _stub_articles(articles: list[dict[str, Any]]) -> None:
    responses.add(
        responses.GET,
        ARTICLE_LIST_URL,
        json={"Data": articles},
    )


def _seed_news_provider(
    engine: Engine,
    coindesk_base_data: dict[str, int],
    *,
    external_code: str,
    name: str = "Source",
) -> int:
    with Session(engine) as session:
        provider = Provider(
            name=name,
            description=name,
            provider_external_code=external_code,
            is_active=True,
            provider_type_id=coindesk_base_data["news_provider_type_id"],
            underlying_provider_id=coindesk_base_data["coindesk_provider_id"],
        )
        session.add(provider)
        session.commit()
        return provider.id


def _materialize(engine: Engine):
    return materialize(
        [coindesk_news_providers, coindesk_news_content],
        selection=AssetSelection.assets(coindesk_news_content),
        resources={
            "postgres": PostgresResource(
                url=engine.url.render_as_string(hide_password=False)
            )
        },
    )


def _article(
    *,
    id: int,
    source_id: int,
    title: str = "Bitcoin breaks $1M",
    body: str = "Optimism abounds as crypto markets surge.",
    authors: str = "Editorial",
    published_on: int | None = None,
) -> dict[str, Any]:
    if published_on is None:
        published_on = int((dt.datetime.now() - dt.timedelta(hours=3)).timestamp())
    return {
        "ID": id,
        "TITLE": title,
        "BODY": body,
        "AUTHORS": authors,
        "PUBLISHED_ON": published_on,
        "SOURCE_ID": source_id,
    }


@responses.activate
def test_inserts_new_content(
    postgres_engine: Engine, coindesk_base_data: dict[str, int]
) -> None:
    source_id = _seed_news_provider(
        postgres_engine, coindesk_base_data, external_code="42"
    )
    _stub_articles(
        [
            _article(id=1001, source_id=42, title="A1", body="positive vibes"),
            _article(id=1002, source_id=42, title="A2", body="market crashes"),
        ]
    )

    result = _materialize(postgres_engine)
    assert result.success

    with Session(postgres_engine) as session:
        rows = session.execute(select(ProviderContent)).scalars().all()
        assert len(rows) == 2
        by_code = {r.content_external_code: r for r in rows}
        assert by_code["1001"].title == "A1"
        assert by_code["1001"].provider_id == source_id
        assert (
            by_code["1001"].content_type_id
            == coindesk_base_data["news_content_type_id"]
        )
        assert by_code["1002"].content == "market crashes"


@responses.activate
def test_drops_articles_from_unmapped_source(
    postgres_engine: Engine, coindesk_base_data: dict[str, int]
) -> None:
    _seed_news_provider(postgres_engine, coindesk_base_data, external_code="42")
    _stub_articles(
        [
            _article(id=2001, source_id=42, title="Mapped"),
            _article(id=2002, source_id=999, title="Unmapped"),  # source 999 not seeded
        ]
    )

    result = _materialize(postgres_engine)
    assert result.success

    with Session(postgres_engine) as session:
        rows = session.execute(select(ProviderContent)).scalars().all()
        assert len(rows) == 1
        assert rows[0].content_external_code == "2001"


@responses.activate
def test_idempotent_on_repeat_run(
    postgres_engine: Engine, coindesk_base_data: dict[str, int]
) -> None:
    _seed_news_provider(postgres_engine, coindesk_base_data, external_code="42")
    payload = [_article(id=3001, source_id=42)]
    _stub_articles(payload)
    assert _materialize(postgres_engine).success

    responses.reset()
    _stub_articles(payload)
    assert _materialize(postgres_engine).success

    with Session(postgres_engine) as session:
        rows = session.execute(select(ProviderContent)).scalars().all()
        assert len(rows) == 1


@responses.activate
def test_updates_existing_content_when_title_changes(
    postgres_engine: Engine, coindesk_base_data: dict[str, int]
) -> None:
    _seed_news_provider(postgres_engine, coindesk_base_data, external_code="42")
    _stub_articles([_article(id=4001, source_id=42, title="OldTitle")])
    assert _materialize(postgres_engine).success

    responses.reset()
    _stub_articles([_article(id=4001, source_id=42, title="NewTitle")])
    assert _materialize(postgres_engine).success

    with Session(postgres_engine) as session:
        row = session.execute(
            select(ProviderContent).where(
                ProviderContent.content_external_code == "4001"
            )
        ).scalar_one()
        assert row.title == "NewTitle"


@responses.activate
def test_empty_api_response_is_no_op(
    postgres_engine: Engine, coindesk_base_data: dict[str, int]
) -> None:
    _seed_news_provider(postgres_engine, coindesk_base_data, external_code="42")
    _stub_articles([])

    assert _materialize(postgres_engine).success

    with Session(postgres_engine) as session:
        rows = session.execute(select(ProviderContent)).scalars().all()
        assert rows == []


def _run_check_only(engine: Engine):
    return materialize(
        [coindesk_news_content, coindesk_news_content_quality],
        selection=AssetSelection.checks(coindesk_news_content_quality),
        resources={
            "postgres": PostgresResource(
                url=engine.url.render_as_string(hide_password=False)
            )
        },
    )


@responses.activate
def test_quality_check_passes_after_materialize(
    postgres_engine: Engine, coindesk_base_data: dict[str, int]
) -> None:
    _seed_news_provider(postgres_engine, coindesk_base_data, external_code="42")
    _stub_articles([_article(id=5001, source_id=42)])
    assert _materialize(postgres_engine).success

    result = _run_check_only(postgres_engine)
    assert result.success
    evals = result.get_asset_check_evaluations()
    assert evals[0].passed is True
    assert evals[0].metadata["rows_in_last_4h"].value == 1


def test_quality_check_fails_when_no_recent_content(
    postgres_engine: Engine, coindesk_base_data: dict[str, int]
) -> None:
    result = _run_check_only(postgres_engine)
    assert result.success
    evals = result.get_asset_check_evaluations()
    assert evals[0].passed is False
    assert "no NEWS content rows in last 4h" in (evals[0].description or "")


def test_quality_check_fails_on_empty_title(
    postgres_engine: Engine, coindesk_base_data: dict[str, int]
) -> None:
    source_id = _seed_news_provider(
        postgres_engine, coindesk_base_data, external_code="42"
    )
    with Session(postgres_engine) as session:
        session.add(
            ProviderContent(
                timestamp=dt.datetime.now() - dt.timedelta(minutes=30),
                provider_id=source_id,
                content_external_code="9000",
                content_type_id=coindesk_base_data["news_content_type_id"],
                authors="x",
                title="",
                content="some content",
            )
        )
        session.commit()

    result = _run_check_only(postgres_engine)
    evals = result.get_asset_check_evaluations()
    assert evals[0].passed is False
    assert "null/empty title" in (evals[0].description or "")
    assert evals[0].metadata["null_title"].value == 1
