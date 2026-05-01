import datetime as dt

from sqlalchemy import Engine, func, select
from sqlalchemy.orm import Session
from mc_postgres_db.models import Asset, Provider, ProviderAsset


def provider_asset_map(
    engine: Engine, provider_name: str, as_of: dt.date
) -> tuple[int, dict[str, int]]:
    with Session(engine) as session:
        provider_id = session.execute(
            select(Provider.id).where(Provider.name == provider_name)
        ).scalar_one()

        subq = (
            select(
                ProviderAsset.asset_code,
                ProviderAsset.provider_id,
                func.max(ProviderAsset.date).label("max_date"),
            )
            .where(ProviderAsset.date <= as_of, ProviderAsset.is_active.is_(True))
            .group_by(ProviderAsset.asset_code, ProviderAsset.provider_id)
            .subquery()
        )
        q = (
            select(ProviderAsset.asset_code, ProviderAsset.asset_id)
            .join(
                subq,
                (ProviderAsset.asset_code == subq.c.asset_code)
                & (ProviderAsset.provider_id == subq.c.provider_id)
                & (ProviderAsset.date == subq.c.max_date),
            )
            .join(Asset, Asset.id == ProviderAsset.asset_id)
            .where(ProviderAsset.provider_id == provider_id, Asset.is_active.is_(True))
        )
        rows = session.execute(q).all()

    return provider_id, {code: aid for code, aid in rows}
