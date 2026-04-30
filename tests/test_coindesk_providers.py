from typing import Any

import responses
from dagster import AssetSelection, materialize
from mc_postgres_db.models import Provider
from sqlalchemy import Engine, select
from sqlalchemy.orm import Session

from dagster_loaders.defs.coindesk.common import COINDESK_API_HOST
from dagster_loaders.defs.coindesk.providers import (
    coindesk_news_providers,
    coindesk_news_providers_quality,
)
from dagster_loaders.resources import PostgresResource


SOURCE_LIST_URL = f"{COINDESK_API_HOST}/news/v1/source/list"


def _stub_source_list(sources: list[dict[str, Any]]) -> None:
    responses.add(
        responses.GET,
        SOURCE_LIST_URL,
        json={"Data": sources},
    )


def _materialize(engine: Engine):
    return materialize(
        [coindesk_news_providers],
        resources={
            "postgres": PostgresResource(
                url=engine.url.render_as_string(hide_password=False)
            )
        },
    )


@responses.activate
def test_inserts_new_providers(
    postgres_engine: Engine, coindesk_base_data: dict[str, int]
) -> None:
    _stub_source_list(
        [
            {
                "ID": 100,
                "NAME": "CryptoSlate",
                "URL": "https://cryptoslate.com",
                "IMAGE_URL": "https://cryptoslate.com/logo.png",
                "STATUS": "ACTIVE",
            },
            {
                "ID": 101,
                "NAME": "The Block",
                "URL": "https://theblock.co",
                "IMAGE_URL": "https://theblock.co/logo.png",
                "STATUS": "ACTIVE",
            },
        ]
    )

    result = _materialize(postgres_engine)
    assert result.success

    with Session(postgres_engine) as session:
        children = (
            session.execute(
                select(Provider).where(
                    Provider.underlying_provider_id
                    == coindesk_base_data["coindesk_provider_id"]
                )
            )
            .scalars()
            .all()
        )
        assert len(children) == 2
        by_code = {p.provider_external_code: p for p in children}
        assert by_code["100"].name == "CryptoSlate"
        assert by_code["100"].is_active is True
        assert (
            by_code["100"].provider_type_id
            == coindesk_base_data["news_provider_type_id"]
        )
        assert by_code["101"].name == "The Block"


@responses.activate
def test_idempotent_on_repeat_run(
    postgres_engine: Engine, coindesk_base_data: dict[str, int]
) -> None:
    payload = [
        {
            "ID": 200,
            "NAME": "Decrypt",
            "URL": "https://decrypt.co",
            "IMAGE_URL": "https://decrypt.co/logo.png",
            "STATUS": "ACTIVE",
        }
    ]
    _stub_source_list(payload)
    assert _materialize(postgres_engine).success

    responses.reset()
    _stub_source_list(payload)
    assert _materialize(postgres_engine).success

    with Session(postgres_engine) as session:
        children = (
            session.execute(
                select(Provider).where(
                    Provider.underlying_provider_id
                    == coindesk_base_data["coindesk_provider_id"]
                )
            )
            .scalars()
            .all()
        )
        assert len(children) == 1
        assert children[0].provider_external_code == "200"


@responses.activate
def test_updates_existing_provider_when_name_changes(
    postgres_engine: Engine, coindesk_base_data: dict[str, int]
) -> None:
    _stub_source_list(
        [
            {
                "ID": 300,
                "NAME": "OldName",
                "URL": "https://old.example",
                "IMAGE_URL": "https://old.example/logo.png",
                "STATUS": "ACTIVE",
            }
        ]
    )
    assert _materialize(postgres_engine).success

    responses.reset()
    _stub_source_list(
        [
            {
                "ID": 300,
                "NAME": "NewName",
                "URL": "https://new.example",
                "IMAGE_URL": "https://new.example/logo.png",
                "STATUS": "ACTIVE",
            }
        ]
    )
    assert _materialize(postgres_engine).success

    with Session(postgres_engine) as session:
        rows = (
            session.execute(
                select(Provider).where(
                    Provider.underlying_provider_id
                    == coindesk_base_data["coindesk_provider_id"]
                )
            )
            .scalars()
            .all()
        )
        assert len(rows) == 1
        assert rows[0].name == "NewName"
        assert rows[0].url == "https://new.example"


@responses.activate
def test_inactive_status_marked_is_active_false(
    postgres_engine: Engine, coindesk_base_data: dict[str, int]
) -> None:
    _stub_source_list(
        [
            {
                "ID": 400,
                "NAME": "Defunct",
                "URL": "https://defunct.example",
                "IMAGE_URL": "https://defunct.example/logo.png",
                "STATUS": "DEPRECATED",
            }
        ]
    )
    assert _materialize(postgres_engine).success

    with Session(postgres_engine) as session:
        row = session.execute(
            select(Provider).where(Provider.provider_external_code == "400")
        ).scalar_one()
        assert row.is_active is False


def _run_check_only(engine: Engine):
    return materialize(
        [coindesk_news_providers, coindesk_news_providers_quality],
        selection=AssetSelection.checks(coindesk_news_providers_quality),
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
    _stub_source_list(
        [
            {
                "ID": 500,
                "NAME": "Bitcoin Magazine",
                "URL": "https://bitcoinmagazine.com",
                "IMAGE_URL": "https://bitcoinmagazine.com/logo.png",
                "STATUS": "ACTIVE",
            }
        ]
    )
    assert _materialize(postgres_engine).success

    result = _run_check_only(postgres_engine)
    assert result.success
    evals = result.get_asset_check_evaluations()
    assert len(evals) == 1
    assert evals[0].passed is True
    assert evals[0].metadata["total"].value == 1


def test_quality_check_fails_when_no_providers(
    postgres_engine: Engine, coindesk_base_data: dict[str, int]
) -> None:
    result = _run_check_only(postgres_engine)
    assert result.success
    evals = result.get_asset_check_evaluations()
    assert evals[0].passed is False
    assert "0 Coindesk news providers" in (evals[0].description or "")


def test_quality_check_fails_on_empty_name(
    postgres_engine: Engine, coindesk_base_data: dict[str, int]
) -> None:
    with Session(postgres_engine) as session:
        session.add(
            Provider(
                provider_external_code="600",
                name="",
                url="https://nameless.example",
                is_active=True,
                provider_type_id=coindesk_base_data["news_provider_type_id"],
                underlying_provider_id=coindesk_base_data["coindesk_provider_id"],
            )
        )
        session.commit()

    result = _run_check_only(postgres_engine)
    evals = result.get_asset_check_evaluations()
    assert evals[0].passed is False
    assert "null/empty name" in (evals[0].description or "")
    assert evals[0].metadata["null_name"].value == 1
