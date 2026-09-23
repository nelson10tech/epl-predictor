from __future__ import annotations

import threading
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Mapping, Optional

from .data import MatchRepository
from .model import DixonColesPredictor, EloPoissonPredictor


APP_VERSION = "2.1"
MODEL_VERSION = "2.0"


@dataclass(frozen=True)
class RuntimeState:
    repository: MatchRepository
    predictors: Mapping[str, object]


def build_predictors(
    repository: MatchRepository, calibration_temperature: float
) -> dict[str, object]:
    return {
        "v1": EloPoissonPredictor(repository.matches),
        "v2": DixonColesPredictor(
            repository.matches,
            calibration_temperature=calibration_temperature,
        ),
    }


class PredictionRuntime:
    """Owns the live immutable state and refreshes it in the background."""

    def __init__(
        self,
        repository: MatchRepository,
        calibration_temperature: float,
        max_age_hours: float = 12.0,
        check_interval_seconds: float = 300.0,
        auto_refresh: bool = True,
    ) -> None:
        self.calibration_temperature = calibration_temperature
        self.max_age_hours = max_age_hours
        self.check_interval = timedelta(seconds=max(check_interval_seconds, 1.0))
        self.auto_refresh = auto_refresh
        self._state = RuntimeState(
            repository=repository,
            predictors=build_predictors(repository, calibration_temperature),
        )
        self._state_lock = threading.RLock()
        self._control_lock = threading.Lock()
        self._refresh_in_progress = False
        self._last_check: Optional[datetime] = datetime.now(timezone.utc)
        self._refresh_thread: Optional[threading.Thread] = None

    def snapshot(self) -> RuntimeState:
        with self._state_lock:
            return self._state

    def freshness_metadata(self) -> dict:
        with self._state_lock:
            metadata = self._state.repository.freshness_metadata()
        with self._control_lock:
            metadata["refresh_in_progress"] = self._refresh_in_progress
        return metadata

    def maybe_refresh(self, now: Optional[datetime] = None) -> bool:
        """Start one stale refresh without delaying the request that noticed it."""
        if not self.auto_refresh:
            return False
        now = now or datetime.now(timezone.utc)
        with self._control_lock:
            if self._refresh_in_progress:
                return False
            if self._last_check and now - self._last_check < self.check_interval:
                return False
            self._last_check = now

        with self._state_lock:
            repository = self._state.repository
            stale = repository.check_stale(self.max_age_hours, now)
            if stale:
                repository.mark_refresh_started(now)
        if not stale:
            return False

        with self._control_lock:
            if self._refresh_in_progress:
                return False
            self._refresh_in_progress = True
            thread = threading.Thread(
                target=self._refresh_worker,
                args=(now,),
                name="epl-data-refresh",
                daemon=True,
            )
            self._refresh_thread = thread
            thread.start()
        return True

    def _refresh_worker(self, started_at: datetime) -> None:
        try:
            new_state = self._create_refreshed_state(started_at)
            with self._state_lock:
                self._state = new_state
        except Exception as exc:
            with self._state_lock:
                self._state.repository.mark_refresh_failure(started_at, str(exc))
        finally:
            with self._control_lock:
                self._refresh_in_progress = False

    def _create_refreshed_state(self, started_at: datetime) -> RuntimeState:
        current = self.snapshot().repository
        candidate = MatchRepository(
            history_seasons=current.history_seasons,
            timeout=current.timeout,
            data_dir=current.data_dir,
        )
        candidate.load()
        candidate.refresh(now=started_at)
        predictors = build_predictors(candidate, self.calibration_temperature)
        return RuntimeState(repository=candidate, predictors=predictors)

    def wait_for_refresh(self, timeout: float = 5.0) -> bool:
        with self._control_lock:
            thread = self._refresh_thread
        if thread is not None:
            thread.join(timeout)
        with self._control_lock:
            return not self._refresh_in_progress

    def reset_check_timer_for_tests(self) -> None:
        with self._control_lock:
            self._last_check = None
