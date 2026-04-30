from dagster import Definitions, EnvVar

from dagster_loaders.resources import PostgresResource


defs = Definitions(
    resources={"postgres": PostgresResource(url=EnvVar("POSTGRES_URL"))},
)
