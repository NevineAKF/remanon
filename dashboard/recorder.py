"""
Band A, Layer L9 — showcase recorder.

Captures a full demo run into a self-contained artifact for static hosting
(GitHub Pages): the complete EventLog stream plus a series of timestamped
/api/state-shaped snapshots, written to dashboard/showcase/run_recording.json
and spliced into dashboard/showcase/index.html's inline data island so the
showcase page needs no server, no fetch, and works from file://.

Read-only with respect to Band B: RunRecorder only ever reads the EventLog,
CoreMetrics, MemoryModel, and ResidencyManager already passed to it.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from dashboard.server import DashboardSources, build_state_snapshot

SHOWCASE_DIR = Path(__file__).parent / "showcase"
RECORDING_PATH = SHOWCASE_DIR / "run_recording.json"
SHOWCASE_HTML_PATH = SHOWCASE_DIR / "index.html"

_DATA_ISLAND_RE = re.compile(
    r'(<script type="application/json" id="run-recording-data">)(.*?)(</script>)',
    re.S,
)


@dataclass
class RunRecorder:
    """
    Tap alongside a live run: call .snapshot() at meaningful moments (boot,
    each verdict, periodically during replay), then .build_recording() once
    the run is done. t0 anchors every timestamp to "seconds since recording
    started" — the same relative timing the showcase driver replays.
    """

    sources: DashboardSources
    t0: datetime = field(default_factory=lambda: datetime.now(UTC))
    _snapshots: list[dict[str, Any]] = field(default_factory=list)

    def snapshot(self) -> None:
        state = build_state_snapshot(self.sources)
        t_rel = (datetime.now(UTC) - self.t0).total_seconds()
        self._snapshots.append({"t_rel": t_rel, "state": state})

    def build_recording(self, *, meta: dict[str, Any]) -> dict[str, Any]:
        """meta is caller-supplied run metadata (speed, burst rule, counts, …)."""
        events = []
        for event in self.sources.event_log.events():
            row = event.to_dict()
            row["t_rel"] = (event.ts - self.t0).total_seconds()
            events.append(row)
        return {"meta": meta, "events": events, "snapshots": self._snapshots}


def write_recording(recording: dict[str, Any], path: Path = RECORDING_PATH) -> Path:
    """Write the standalone, human-inspectable recording JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(recording, separators=(",", ":")), encoding="utf-8")
    return path


def embed_recording_in_showcase(
    recording: dict[str, Any], html_path: Path = SHOWCASE_HTML_PATH
) -> Path:
    """
    Splice *recording* into the showcase page's inline data island, so the
    page is self-contained: no fetch, works from file:// with no server.
    """
    if not html_path.exists():
        raise FileNotFoundError(
            f"{html_path} does not exist — the showcase page template must be "
            "created before a recording can be embedded into it."
        )
    html = html_path.read_text(encoding="utf-8")
    if not _DATA_ISLAND_RE.search(html):
        raise ValueError(
            f"{html_path} has no "
            '<script type="application/json" id="run-recording-data"> data island '
            "to embed the recording into."
        )
    # Escape "</" so no recorded string can prematurely close the <script> tag.
    payload = json.dumps(recording, separators=(",", ":")).replace("</", "<\\/")
    new_html = _DATA_ISLAND_RE.sub(lambda m: m.group(1) + payload + m.group(3), html, count=1)
    html_path.write_text(new_html, encoding="utf-8")
    return html_path
