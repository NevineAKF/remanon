"""
Band A, Layer L6 — master-context digest.

DigestBuilder condenses the TelemetryStore into a compact string via SQL:
node list, level distribution, time range, top components. This digest is
the master context that gets materialized and pinned at boot.
"""

from __future__ import annotations

from app.dataplane.store import TelemetryStore

# The digest is sent verbatim as Contract B system-message content when
# pinning each master (core/materializer.py::_prefill) — a full node
# enumeration on a large real store (thousands of distinct nodes) produces a
# multi-tens-of-KB payload that real inference engines can reject (context
# length / chat-template limits). Only the node list is capped: it's the one
# field whose size scales with store size rather than staying summary-sized.
_MAX_NODES_LISTED = 50
_MAX_DIGEST_CHARS = 4000


class DigestBuilder:
    def __init__(self, store: TelemetryStore) -> None:
        self._store = store

    def build(self, top_components: int = 5) -> str:
        store = self._store
        total = store.count()
        if total == 0:
            return "TELEMETRY DIGEST\nrecords=0 (empty store)"

        span = store.time_range()
        assert span is not None
        lo, hi = span

        nodes = [
            r["node"] for r in store.query("SELECT DISTINCT node FROM telemetry ORDER BY node")
        ]
        levels = store.query(
            "SELECT level, COUNT(*) AS n FROM telemetry GROUP BY level ORDER BY n DESC, level"
        )
        components = store.query(
            "SELECT component, COUNT(*) AS n FROM telemetry "
            "GROUP BY component ORDER BY n DESC, component "
            f"LIMIT {int(top_components)}"
        )

        if len(nodes) > _MAX_NODES_LISTED:
            shown = nodes[:_MAX_NODES_LISTED]
            nodes_line = (
                "nodes=" + ", ".join(shown) + f", ...and {len(nodes) - _MAX_NODES_LISTED} more"
            )
        else:
            nodes_line = "nodes=" + ", ".join(nodes)

        digest = "\n".join(
            [
                "TELEMETRY DIGEST",
                f"records={total}",
                f"span={lo.isoformat()} .. {hi.isoformat()}",
                nodes_line,
                "levels=" + ", ".join(f"{r['level']}:{r['n']}" for r in levels),
                "top_components=" + ", ".join(f"{r['component']}:{r['n']}" for r in components),
            ]
        )
        if len(digest) > _MAX_DIGEST_CHARS:
            digest = digest[: _MAX_DIGEST_CHARS - 3] + "..."
        return digest
