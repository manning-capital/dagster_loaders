from dagster import EnvVar, Definitions

from dagster_loaders.resources import PostgresResource

defs = Definitions(
    resources={"postgres": PostgresResource(url=EnvVar("POSTGRES_URL"))},
)
