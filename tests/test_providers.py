from __future__ import annotations

import json
from datetime import datetime, timezone

from app.data import MatchRepository
from app.feature_store import ChronologicalFeatureStore
from app.providers import (
    NullAllCompetitionScheduleProvider,
    NullPlayerAvailabilityProvider,
    OpenFootballFixtureProvider,
)
from app.xg import NullXGProvider


def test_fixture_provider_failure_uses_cache(tmp_path, monkeypatch):
    cache = tmp_path / "fixtures.json"
    cache.write_text(json.dumps({
        "source_url": "https://example.test/feed.json",
        "fetched_at": "2026-09-23T00:00:00+00:00",
        "fixtures": [{
            "date": "2026-10-10",
            "season": "2627",
            "kickoff": "15:00",
            "home_team": "Arsenal",
            "away_team": "Chelsea",
        }],
    }))
    provider = OpenFootballFixtureProvider(cache_path=cache)

    def fail(*args, **kwargs):
        raise RuntimeError("fixture source unavailable")

    monkeypatch.setattr("app.providers.requests.get", fail)
    fixtures = provider.fetch_upcoming(
        now=datetime(2026, 9, 23, tzinfo=timezone.utc),
        known_teams={"Arsenal", "Chelsea"},
    )
    assert len(fixtures) == 1
    assert fixtures[0].home_team == "Arsenal"


def test_missing_optional_sources_do_not_crash_feature_store():
    repository = MatchRepository()
    repository.load()
    store = ChronologicalFeatureStore(
        repository.matches[:500],
        xg_provider=NullXGProvider(),
        schedule_provider=NullAllCompetitionScheduleProvider(),
        player_provider=NullPlayerAvailabilityProvider(),
    )
    assert len(store.rows) == 500
    assert store.metadata()["xg"]["enabled"] is False
    assert store.metadata()["all_competition_schedule"]["congestion_scope"] == "EPL only"
    assert store.metadata()["player_availability"]["available"] is False
