import io
import pathlib
import zipfile
import datetime as dt
from typing import Any

import pytest
import responses
from dagster import Failure, materialize, build_asset_context
from sqlalchemy import Engine, select
from sqlalchemy.orm import Session
from mc_postgres_db.models import ProviderAssetMarket

from dagster_loaders.resources import PostgresResource
from dagster_loaders.defs.kraken import historical_market_data
from dagster_loaders.defs.kraken.market_data import ASSET_PAIRS_URL
from dagster_loaders.defs.kraken.historical_market_data import (
    SA_JSON_ENV_VAR,
    _materialize_historical,
    kraken_provider_asset_market_historical,
)


def _make_zip(
    rows_by_pair: dict[str, list[list[Any]]],
    *,
    extra_files: dict[str, list[list[Any]]] | None = None,
) -> bytes:
    """Build an in-memory zip with one `{altname}_1.csv` per `rows_by_pair`
    entry plus arbitrary `extra_files` (e.g. `XBTUSD_60.csv`) for the timeframe
    filter test."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for altname, rows in rows_by_pair.items():
            csv = "\n".join(",".join(str(c) for c in r) for r in rows)
            zf.writestr(f"{altname}_1.csv", csv)
        for name, rows in (extra_files or {}).items():
            csv = "\n".join(",".join(str(c) for c in r) for r in rows)
            zf.writestr(name, csv)
    return buf.getvalue()


def _stub_asset_pairs(pairs: dict[str, dict[str, Any]]) -> None:
    """`pairs` maps altname -> info dict; helper auto-fills `altname` and
    defaults `execution_venue` to international."""
    result: dict[str, dict[str, Any]] = {}
    for altname, info in pairs.items():
        full = {"altname": altname, "execution_venue": "international", **info}
        result[altname] = full
    responses.add(responses.GET, ASSET_PAIRS_URL, json={"result": result, "error": []})


def _patch_drive(
    monkeypatch: pytest.MonkeyPatch,
    *,
    available_quarters: set[tuple[int, int]] = frozenset(),
    zip_bytes_by_label: dict[str, bytes] | None = None,
) -> dict[str, list[str]]:
    """Patch all Drive interactions. Returns a dict containing a `fetch_calls`
    list which records the label of each zip download in order."""
    if zip_bytes_by_label is None:
        zip_bytes_by_label = {}

    monkeypatch.setenv(
        SA_JSON_ENV_VAR, '{"type":"service_account","project_id":"test"}'
    )
    monkeypatch.setattr(
        historical_market_data, "_drive_service", lambda _info: object()
    )
    monkeypatch.setattr(
        historical_market_data,
        "_list_available_quarters",
        lambda _service, _folder: set(available_quarters),
    )

    fetch_calls: list[str] = []

    def _fake_resolve_zip_path(
        _service: object,
        zip_key: tuple[str, str],
        dest_dir: pathlib.Path,
    ) -> tuple[pathlib.Path, str]:
        kind, ref = zip_key
        if kind == "quarterly":
            label = ref
            filename = ref
        else:
            label = "full_history"
            filename = "kraken_full_history.zip"
        if label not in zip_bytes_by_label:
            raise FileNotFoundError(f"no fixture bytes staged for label={label}")
        path = dest_dir / filename
        path.write_bytes(zip_bytes_by_label[label])
        fetch_calls.append(label)
        return path, label

    monkeypatch.setattr(
        historical_market_data, "_resolve_zip_path", _fake_resolve_zip_path
    )
    return {"fetch_calls": fetch_calls}


def _materialize(engine: Engine, partition_key: str = "2024-01-15"):
    return materialize(
        [kraken_provider_asset_market_historical],
        partition_key=partition_key,
        resources={
            "postgres": PostgresResource(
                url=engine.url.render_as_string(hide_password=False)
            )
        },
        raise_on_error=False,
    )


@responses.activate
def test_materialize_writes_partition_rows_into_empty_db(
    monkeypatch: pytest.MonkeyPatch,
    postgres_engine: Engine,
    kraken_base_data: dict[str, Any],
) -> None:
    base = int(dt.datetime(2024, 1, 15, 12, 0, tzinfo=dt.timezone.utc).timestamp())
    rows = [[base + 60 * i, "100", "100", "100", "100", "100", 1] for i in range(3)]
    _stub_asset_pairs({"XBTUSD": {"base": "XXBT", "quote": "ZUSD"}})
    _patch_drive(
        monkeypatch,
        available_quarters={(2024, 1)},
        zip_bytes_by_label={"Kraken_OHLCVT_Q1_2024.zip": _make_zip({"XBTUSD": rows})},
    )

    assert _materialize(postgres_engine).success

    with Session(postgres_engine) as session:
        all_rows = session.execute(select(ProviderAssetMarket)).scalars().all()
        assert len(all_rows) == 3

        # API loader orientation: from = quote (USD), to = base (BTC).
        usd_btc_rows = (
            session.execute(
                select(ProviderAssetMarket).where(
                    ProviderAssetMarket.from_asset_id
                    == kraken_base_data["usd_asset_id"],
                    ProviderAssetMarket.to_asset_id == kraken_base_data["btc_asset_id"],
                )
            )
            .scalars()
            .all()
        )
        assert len(usd_btc_rows) == 3
        assert all(r.close == 100.0 for r in usd_btc_rows)
        assert all(
            r.provider_id == kraken_base_data["provider_id"] for r in usd_btc_rows
        )


@responses.activate
def test_skips_non_one_minute_csvs(
    monkeypatch: pytest.MonkeyPatch,
    postgres_engine: Engine,
    kraken_base_data: dict[str, Any],
) -> None:
    base = int(dt.datetime(2024, 1, 15, 12, 0, tzinfo=dt.timezone.utc).timestamp())
    one_min = [[base + 60 * i, "100", "100", "100", "100", "100", 1] for i in range(2)]
    sixty_min = [
        [base + 3600 * i, "999", "999", "999", "999", "999", 1] for i in range(5)
    ]
    _stub_asset_pairs({"XBTUSD": {"base": "XXBT", "quote": "ZUSD"}})
    _patch_drive(
        monkeypatch,
        available_quarters={(2024, 1)},
        zip_bytes_by_label={
            "Kraken_OHLCVT_Q1_2024.zip": _make_zip(
                {"XBTUSD": one_min}, extra_files={"XBTUSD_60.csv": sixty_min}
            )
        },
    )

    assert _materialize(postgres_engine).success

    with Session(postgres_engine) as session:
        rows = session.execute(select(ProviderAssetMarket)).scalars().all()
        assert len(rows) == 2
        assert all(r.close == 100.0 for r in rows)


@responses.activate
def test_skips_pair_not_in_asset_map(
    monkeypatch: pytest.MonkeyPatch,
    postgres_engine: Engine,
    kraken_base_data: dict[str, Any],
) -> None:
    """Skipped pair plus a kept pair on the same partition: skipped pair logs a
    [skip] line and writes nothing; kept pair writes its rows."""
    base = int(dt.datetime(2024, 1, 15, 12, 0, tzinfo=dt.timezone.utc).timestamp())
    btc_rows = [[base + 60 * i, "100", "100", "100", "100", "100", 1] for i in range(2)]
    doge_rows = [[base, "1", "1", "1", "1", "1", 1]]
    _stub_asset_pairs(
        {
            "XBTUSD": {"base": "XXBT", "quote": "ZUSD"},
            "DOGEUSD": {"base": "DOGE", "quote": "ZUSD"},
        }
    )
    _patch_drive(
        monkeypatch,
        available_quarters={(2024, 1)},
        zip_bytes_by_label={
            "Kraken_OHLCVT_Q1_2024.zip": _make_zip(
                {"XBTUSD": btc_rows, "DOGEUSD": doge_rows}
            )
        },
    )

    result = _materialize(postgres_engine)
    assert result.success

    with Session(postgres_engine) as session:
        rows = session.execute(select(ProviderAssetMarket)).scalars().all()
        # Only XBTUSD rows; DOGE skipped because DOGE isn't in kraken_base_data.
        assert len(rows) == 2
        assert all(r.to_asset_id == kraken_base_data["btc_asset_id"] for r in rows)


@responses.activate
def test_filters_non_international_venue(
    monkeypatch: pytest.MonkeyPatch,
    postgres_engine: Engine,
    kraken_base_data: dict[str, Any],
) -> None:
    """A non-international XBTUSD plus an international ETHUSD: bitnomial XBT
    is filtered out by the altname map; the partition still has data via ETH."""
    base = int(dt.datetime(2024, 1, 15, 12, 0, tzinfo=dt.timezone.utc).timestamp())
    rows = [[base, "100", "100", "100", "100", "100", 1]]
    _stub_asset_pairs(
        {
            "XBTUSD": {
                "base": "XXBT",
                "quote": "ZUSD",
                "execution_venue": "bitnomial_exchange",
            },
            "ETHUSD": {"base": "XETH", "quote": "ZUSD"},
        }
    )
    _patch_drive(
        monkeypatch,
        available_quarters={(2024, 1)},
        zip_bytes_by_label={
            "Kraken_OHLCVT_Q1_2024.zip": _make_zip({"XBTUSD": rows, "ETHUSD": rows})
        },
    )

    assert _materialize(postgres_engine).success

    with Session(postgres_engine) as session:
        rows_db = session.execute(select(ProviderAssetMarket)).scalars().all()
        assert len(rows_db) == 1
        assert rows_db[0].to_asset_id == kraken_base_data["eth_asset_id"]


@responses.activate
def test_upsert_overwrites_existing(
    monkeypatch: pytest.MonkeyPatch,
    postgres_engine: Engine,
    kraken_base_data: dict[str, Any],
) -> None:
    base = int(dt.datetime(2024, 1, 15, 12, 0, tzinfo=dt.timezone.utc).timestamp())
    rows_v1 = [[base, "100", "100", "100", "100", "100", 1]]
    _stub_asset_pairs({"XBTUSD": {"base": "XXBT", "quote": "ZUSD"}})
    _patch_drive(
        monkeypatch,
        available_quarters={(2024, 1)},
        zip_bytes_by_label={
            "Kraken_OHLCVT_Q1_2024.zip": _make_zip({"XBTUSD": rows_v1})
        },
    )
    assert _materialize(postgres_engine).success

    responses.reset()
    rows_v2 = [[base, "999", "999", "999", "999", "999", 1]]
    _stub_asset_pairs({"XBTUSD": {"base": "XXBT", "quote": "ZUSD"}})
    _patch_drive(
        monkeypatch,
        available_quarters={(2024, 1)},
        zip_bytes_by_label={
            "Kraken_OHLCVT_Q1_2024.zip": _make_zip({"XBTUSD": rows_v2})
        },
    )
    assert _materialize(postgres_engine).success

    with Session(postgres_engine) as session:
        rows = session.execute(select(ProviderAssetMarket)).scalars().all()
        assert len(rows) == 1
        assert rows[0].close == 999.0
        assert rows[0].volume == 999.0


@pytest.mark.parametrize("batch_size", [1, 3, 100])
@responses.activate
def test_batches_large_dataset(
    monkeypatch: pytest.MonkeyPatch,
    postgres_engine: Engine,
    kraken_base_data: dict[str, Any],
    batch_size: int,
) -> None:
    monkeypatch.setattr(historical_market_data, "BATCH_SIZE", batch_size)

    base = int(dt.datetime(2024, 1, 15, 12, 0, tzinfo=dt.timezone.utc).timestamp())
    rows = [[base + 60 * i, "1", "1", "1", "1", "1", 1] for i in range(5)]
    _stub_asset_pairs({"XBTUSD": {"base": "XXBT", "quote": "ZUSD"}})
    _patch_drive(
        monkeypatch,
        available_quarters={(2024, 1)},
        zip_bytes_by_label={"Kraken_OHLCVT_Q1_2024.zip": _make_zip({"XBTUSD": rows})},
    )

    assert _materialize(postgres_engine).success

    with Session(postgres_engine) as session:
        assert len(session.execute(select(ProviderAssetMarket)).scalars().all()) == 5


@responses.activate
def test_prefers_quarterly_when_listed(
    monkeypatch: pytest.MonkeyPatch,
    postgres_engine: Engine,
    kraken_base_data: dict[str, Any],
) -> None:
    base = int(dt.datetime(2024, 1, 15, 12, 0, tzinfo=dt.timezone.utc).timestamp())
    rows = [[base, "100", "100", "100", "100", "100", 1]]
    _stub_asset_pairs({"XBTUSD": {"base": "XXBT", "quote": "ZUSD"}})
    handles = _patch_drive(
        monkeypatch,
        available_quarters={(2024, 1)},
        zip_bytes_by_label={
            "Kraken_OHLCVT_Q1_2024.zip": _make_zip({"XBTUSD": rows}),
            # full_history fixture also staged but should not be touched.
            "full_history": _make_zip({"XBTUSD": rows}),
        },
    )

    assert _materialize(postgres_engine, partition_key="2024-01-15").success
    assert handles["fetch_calls"] == ["Kraken_OHLCVT_Q1_2024.zip"]


@responses.activate
def test_falls_back_to_full_history_when_quarter_missing(
    monkeypatch: pytest.MonkeyPatch,
    postgres_engine: Engine,
    kraken_base_data: dict[str, Any],
) -> None:
    base = int(dt.datetime(2022, 6, 15, 12, 0, tzinfo=dt.timezone.utc).timestamp())
    rows = [[base, "100", "100", "100", "100", "100", 1]]
    _stub_asset_pairs({"XBTUSD": {"base": "XXBT", "quote": "ZUSD"}})
    handles = _patch_drive(
        monkeypatch,
        available_quarters=set(),
        zip_bytes_by_label={"full_history": _make_zip({"XBTUSD": rows})},
    )

    assert _materialize(postgres_engine, partition_key="2022-06-15").success
    assert handles["fetch_calls"] == ["full_history"]

    with Session(postgres_engine) as session:
        rows_db = session.execute(select(ProviderAssetMarket)).scalars().all()
        assert len(rows_db) == 1


@responses.activate
def test_errors_when_partition_has_no_data(
    monkeypatch: pytest.MonkeyPatch,
    postgres_engine: Engine,
    kraken_base_data: dict[str, Any],
) -> None:
    """A partition with zero matching rows raises Failure listing that date.
    Drives `_materialize_historical` directly to bypass the asset's
    RetryPolicy (which would otherwise sleep through 3 retries)."""
    base = int(dt.datetime(2024, 1, 15, 12, 0, tzinfo=dt.timezone.utc).timestamp())
    rows = [[base, "1", "1", "1", "1", "1", 1]]
    _stub_asset_pairs({"DOGEUSD": {"base": "DOGE", "quote": "ZUSD"}})
    _patch_drive(
        monkeypatch,
        available_quarters={(2024, 1)},
        zip_bytes_by_label={"Kraken_OHLCVT_Q1_2024.zip": _make_zip({"DOGEUSD": rows})},
    )

    postgres = PostgresResource(
        url=postgres_engine.url.render_as_string(hide_password=False)
    )
    context = build_asset_context()
    yielded: list[tuple[dt.date, dict]] = []
    with pytest.raises(Failure) as exc_info:
        for ev in _materialize_historical(context, postgres, {dt.date(2024, 1, 15)}):
            yielded.append(ev)
    assert yielded == []
    assert "2024-01-15" in str(exc_info.value)

    with Session(postgres_engine) as session:
        assert session.execute(select(ProviderAssetMarket)).scalars().all() == []


@responses.activate
def test_partial_progress_yields_then_fails_on_empty_partition(
    monkeypatch: pytest.MonkeyPatch,
    postgres_engine: Engine,
    kraken_base_data: dict[str, Any],
) -> None:
    """Two-day backfill: day A has rows (full_history), day B has none
    (quarterly). Day A must materialize successfully; the run must then raise
    Failure listing day B."""
    a = dt.date(2022, 6, 15)
    b = dt.date(2024, 1, 15)
    base_a = int(dt.datetime(2022, 6, 15, 12, 0, tzinfo=dt.timezone.utc).timestamp())
    rows_a = [[base_a, "100", "100", "100", "100", "100", 1]]
    # Quarterly zip exists but has no pairs that survive mapping.
    rows_doge = [
        [
            int(dt.datetime(2024, 1, 15, 12, 0, tzinfo=dt.timezone.utc).timestamp()),
            "1",
            "1",
            "1",
            "1",
            "1",
            1,
        ]
    ]
    _stub_asset_pairs(
        {
            "XBTUSD": {"base": "XXBT", "quote": "ZUSD"},
            "DOGEUSD": {"base": "DOGE", "quote": "ZUSD"},
        }
    )
    _patch_drive(
        monkeypatch,
        available_quarters={(2024, 1)},
        zip_bytes_by_label={
            "full_history": _make_zip({"XBTUSD": rows_a}),
            "Kraken_OHLCVT_Q1_2024.zip": _make_zip({"DOGEUSD": rows_doge}),
        },
    )

    engine = postgres_engine
    postgres = PostgresResource(url=engine.url.render_as_string(hide_password=False))
    context = build_asset_context()

    yielded: list[tuple[dt.date, dict]] = []
    with pytest.raises(Failure) as exc_info:
        for ev in _materialize_historical(context, postgres, {a, b}):
            yielded.append(ev)

    assert [date for date, _ in yielded] == [a]
    assert b.isoformat() in str(exc_info.value)

    with Session(postgres_engine) as session:
        rows = session.execute(select(ProviderAssetMarket)).scalars().all()
        assert len(rows) == 1


@responses.activate
def test_single_run_backfill_spans_quarterly_and_full_history(
    monkeypatch: pytest.MonkeyPatch,
    postgres_engine: Engine,
    kraken_base_data: dict[str, Any],
) -> None:
    """Two dates spanning a covered quarter and an uncovered date: each zip
    is fetched exactly once and the right rows land under the right date."""
    pre = dt.date(2022, 6, 15)
    post = dt.date(2024, 1, 15)
    pre_ts = int(dt.datetime(2022, 6, 15, 12, 0, tzinfo=dt.timezone.utc).timestamp())
    post_ts = int(dt.datetime(2024, 1, 15, 12, 0, tzinfo=dt.timezone.utc).timestamp())

    _stub_asset_pairs({"XBTUSD": {"base": "XXBT", "quote": "ZUSD"}})
    handles = _patch_drive(
        monkeypatch,
        available_quarters={(2024, 1)},
        zip_bytes_by_label={
            "full_history": _make_zip(
                {"XBTUSD": [[pre_ts, "1", "1", "1", "1", "1", 1]]}
            ),
            "Kraken_OHLCVT_Q1_2024.zip": _make_zip(
                {"XBTUSD": [[post_ts, "2", "2", "2", "2", "2", 1]]}
            ),
        },
    )

    postgres = PostgresResource(
        url=postgres_engine.url.render_as_string(hide_password=False)
    )
    context = build_asset_context()
    yielded = list(_materialize_historical(context, postgres, {pre, post}))

    # Each zip fetched exactly once.
    assert sorted(handles["fetch_calls"]) == sorted(
        ["Kraken_OHLCVT_Q1_2024.zip", "full_history"]
    )
    assert len(handles["fetch_calls"]) == 2
    # One yielded result per requested date.
    assert {date for date, _ in yielded} == {pre, post}

    with Session(postgres_engine) as session:
        rows = session.execute(select(ProviderAssetMarket)).scalars().all()
        assert len(rows) == 2
        closes_by_date = {r.timestamp.date(): r.close for r in rows}
        assert closes_by_date[pre] == 1.0
        assert closes_by_date[post] == 2.0


@responses.activate
def test_yields_partition_results_incrementally_per_zip(
    monkeypatch: pytest.MonkeyPatch,
    postgres_engine: Engine,
    kraken_base_data: dict[str, Any],
) -> None:
    """Across two zips, a date assigned to the first zip yields its result
    before the second zip is fetched."""
    pre = dt.date(2022, 6, 15)
    post = dt.date(2024, 1, 15)
    pre_ts = int(dt.datetime(2022, 6, 15, 12, 0, tzinfo=dt.timezone.utc).timestamp())
    post_ts = int(dt.datetime(2024, 1, 15, 12, 0, tzinfo=dt.timezone.utc).timestamp())

    _stub_asset_pairs({"XBTUSD": {"base": "XXBT", "quote": "ZUSD"}})
    handles = _patch_drive(
        monkeypatch,
        available_quarters={(2024, 1)},
        zip_bytes_by_label={
            "full_history": _make_zip(
                {"XBTUSD": [[pre_ts, "1", "1", "1", "1", "1", 1]]}
            ),
            "Kraken_OHLCVT_Q1_2024.zip": _make_zip(
                {"XBTUSD": [[post_ts, "2", "2", "2", "2", "2", 1]]}
            ),
        },
    )

    postgres = PostgresResource(
        url=postgres_engine.url.render_as_string(hide_password=False)
    )
    context = build_asset_context()

    fetches_when_yielded: list[tuple[str, list[str]]] = []
    for date, _metadata in _materialize_historical(context, postgres, {pre, post}):
        fetches_when_yielded.append((date.isoformat(), list(handles["fetch_calls"])))

    # Each yielded partition has all rows from its zip already in the DB at
    # yield time AND the next zip has not yet been fetched.
    by_partition = dict(fetches_when_yielded)
    # The first zip in `fetch_calls` covers a single date that yields before
    # the second zip is fetched.
    first_zip = handles["fetch_calls"][0]
    second_zip = handles["fetch_calls"][1]
    first_date_key = (
        pre.isoformat() if first_zip == "full_history" else post.isoformat()
    )
    second_date_key = (
        post.isoformat() if first_zip == "full_history" else pre.isoformat()
    )
    assert by_partition[first_date_key] == [first_zip]
    assert by_partition[second_date_key] == [first_zip, second_zip]
