import datetime as dt
import logging
import os
import socket
import time
import uuid
from collections.abc import Generator
from typing import Any

import docker
import pytest
from mc_postgres_db.models import (
    Asset,
    AssetType,
    Base,
    ContentType,
    Provider,
    ProviderAsset,
    ProviderType,
    SentimentType,
)
from mc_postgres_db.testing.utilities import (
    TEST_DB_NAME,
    TEST_DB_PASSWORD,
    TEST_DB_USER,
    clear_database,
)
from sqlalchemy import Engine, create_engine, select, text
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session

LOGGER = logging.getLogger(__name__)


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        s.listen(1)
        return s.getsockname()[1]


def _wait_for_postgres(url: str, timeout: int = 30) -> None:
    start = time.time()
    while time.time() - start < timeout:
        try:
            engine = create_engine(url)
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            engine.dispose()
            return
        except OperationalError:
            time.sleep(1)
    raise TimeoutError(f"Postgres did not become ready within {timeout}s")


@pytest.fixture(scope="session")
def postgres_engine() -> Generator[Engine, None, None]:
    client = docker.from_env()
    unique_id = uuid.uuid4().hex[:8]
    container_name = f"dagster-loaders-test-{unique_id}"
    port = _find_free_port()
    image = f"postgres:{os.getenv('POSTGRES_VERSION', 'latest')}"

    container = client.containers.run(
        image,
        name=container_name,
        environment={
            "POSTGRES_USER": TEST_DB_USER,
            "POSTGRES_PASSWORD": TEST_DB_PASSWORD,
            "POSTGRES_DB": TEST_DB_NAME,
        },
        ports={5432: port},
        detach=True,
        remove=False,
    )

    url = f"postgresql://{TEST_DB_USER}:{TEST_DB_PASSWORD}@localhost:{port}/{TEST_DB_NAME}"
    engine: Engine | None = None
    try:
        _wait_for_postgres(url)
        engine = create_engine(url)
        Base.metadata.create_all(engine)
        yield engine
    finally:
        if engine is not None:
            try:
                Base.metadata.drop_all(engine)
            except Exception as e:
                LOGGER.warning(f"drop_all failed: {e}")
            engine.dispose()
        try:
            container.stop(timeout=10)
            container.remove(v=True)
        except Exception as e:
            LOGGER.warning(f"container teardown failed: {e}")


@pytest.fixture(autouse=True)
def clean_db(postgres_engine: Engine) -> Generator[Engine, None, None]:
    clear_database(postgres_engine)
    yield postgres_engine
    clear_database(postgres_engine)


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "dagster_loaders.defs.kraken_market_data.time.sleep", lambda *_: None
    )


@pytest.fixture
def kraken_base_data(postgres_engine: Engine) -> dict[str, Any]:
    """Seed the Kraken provider plus four assets and provider-asset mappings.

    Returns a dict of ids: provider_id, btc/eth/usd/one_inch asset_ids.
    """
    yesterday = (dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=1)).date()
    with Session(postgres_engine) as session:
        provider_type = ProviderType(
            name="CryptoCurrencyExchange", description="CryptoCurrencyExchange"
        )
        session.add(provider_type)
        session.commit()

        provider = Provider(
            name="Kraken", description="Kraken", provider_type_id=provider_type.id
        )
        session.add(provider)
        session.commit()

        crypto_type = AssetType(name="CryptoCurrency", description="CryptoCurrency")
        fiat_type = AssetType(name="FiatCurrency", description="FiatCurrency")
        session.add_all([crypto_type, fiat_type])
        session.commit()

        btc = Asset(name="BTC", description="BTC", asset_type_id=crypto_type.id)
        eth = Asset(name="ETH", description="ETH", asset_type_id=crypto_type.id)
        usd = Asset(name="USD", description="USD", asset_type_id=fiat_type.id)
        one_inch = Asset(
            name="1INCH", description="1INCH", asset_type_id=crypto_type.id
        )
        session.add_all([btc, eth, usd, one_inch])
        session.commit()

        session.add_all(
            [
                ProviderAsset(
                    date=yesterday,
                    provider_id=provider.id,
                    asset_id=btc.id,
                    asset_code="XXBT",
                    is_active=True,
                ),
                ProviderAsset(
                    date=yesterday,
                    provider_id=provider.id,
                    asset_id=eth.id,
                    asset_code="XETH",
                    is_active=True,
                ),
                ProviderAsset(
                    date=yesterday,
                    provider_id=provider.id,
                    asset_id=usd.id,
                    asset_code="ZUSD",
                    is_active=True,
                ),
                ProviderAsset(
                    date=yesterday,
                    provider_id=provider.id,
                    asset_id=one_inch.id,
                    asset_code="1INCH",
                    is_active=True,
                ),
            ]
        )
        session.commit()

        return {
            "provider_id": session.execute(
                select(Provider.id).where(Provider.name == "Kraken")
            ).scalar_one(),
            "btc_asset_id": session.execute(
                select(Asset.id).where(Asset.name == "BTC")
            ).scalar_one(),
            "eth_asset_id": session.execute(
                select(Asset.id).where(Asset.name == "ETH")
            ).scalar_one(),
            "usd_asset_id": session.execute(
                select(Asset.id).where(Asset.name == "USD")
            ).scalar_one(),
            "one_inch_asset_id": session.execute(
                select(Asset.id).where(Asset.name == "1INCH")
            ).scalar_one(),
        }


@pytest.fixture
def coindesk_base_data(postgres_engine: Engine) -> dict[str, int]:
    """Seed the COINDESK parent provider, NEWS_PROVIDER type, NEWS content type,
    and NLTKVader sentiment type. Returns a dict of ids."""
    with Session(postgres_engine) as session:
        news_provider_type = ProviderType(
            name="NEWS_PROVIDER", description="news provider"
        )
        session.add(news_provider_type)
        session.commit()

        coindesk = Provider(
            name="Coindesk",
            description="Coindesk",
            provider_external_code="COINDESK",
            provider_type_id=news_provider_type.id,
        )
        session.add(coindesk)
        session.commit()

        news_content_type = ContentType(name="NEWS", description="news article")
        session.add(news_content_type)
        session.commit()

        nltk_vader = SentimentType(
            name="NLTKVader", description="NLTK VADER compound sentiment"
        )
        session.add(nltk_vader)
        session.commit()

        return {
            "coindesk_provider_id": coindesk.id,
            "news_provider_type_id": news_provider_type.id,
            "news_content_type_id": news_content_type.id,
            "nltk_vader_sentiment_type_id": nltk_vader.id,
        }
