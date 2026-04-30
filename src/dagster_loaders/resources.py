from dagster import ConfigurableResource
from sqlalchemy import Engine, create_engine


class PostgresResource(ConfigurableResource):
    url: str

    def get_engine(self) -> Engine:
        return create_engine(self.url)
