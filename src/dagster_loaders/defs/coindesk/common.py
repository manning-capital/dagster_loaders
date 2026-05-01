import datetime as dt

from dagster import Backoff, RetryPolicy, FreshnessPolicy

COINDESK_API_HOST = "https://data-api.coindesk.com"
COINDESK_API_POOL = "coindesk-api"
COINDESK_SENTIMENT_POOL = "coindesk-sentiment"

CONTENT_LOOKBACK = dt.timedelta(hours=2)
RECENT_CONTENT_WINDOW = dt.timedelta(hours=4)

PROVIDER_COLUMNS = [
    "id",
    "provider_external_code",
    "name",
    "url",
    "image_url",
    "is_active",
    "provider_type_id",
    "underlying_provider_id",
]
CONTENT_COLUMNS = [
    "id",
    "timestamp",
    "provider_id",
    "content_external_code",
    "content_type_id",
    "authors",
    "title",
    "content",
]

FRESHNESS = FreshnessPolicy.time_window(
    fail_window=dt.timedelta(minutes=120),
    warn_window=dt.timedelta(minutes=90),
)
RETRY = RetryPolicy(max_retries=3, delay=5.0, backoff=Backoff.EXPONENTIAL)
