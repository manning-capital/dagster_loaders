from dagster import (
    Definitions,
    AssetSelection,
    ScheduleDefinition,
    define_asset_job,
)

from dagster_loaders.defs.coindesk.content import coindesk_news_content
from dagster_loaders.defs.coindesk.providers import coindesk_news_providers
from dagster_loaders.defs.coindesk.sentiment import coindesk_content_sentiment

coindesk_content_job = define_asset_job(
    name="coindesk_content_job",
    selection=AssetSelection.assets(
        coindesk_news_providers,
        coindesk_news_content,
        coindesk_content_sentiment,
    ),
)

coindesk_content_schedule = ScheduleDefinition(
    name="coindesk_content_hourly",
    cron_schedule="0 * * * *",
    job=coindesk_content_job,
)


defs = Definitions(
    jobs=[coindesk_content_job],
    schedules=[coindesk_content_schedule],
)
