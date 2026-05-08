"""SkullChain — tamper-evident local audit trail.

Each block is a SHA-256 hash of (prev_hash ‖ iso_timestamp ‖ username ‖
patient_hash ‖ action ‖ payload_hash).  Altering any block invalidates
all subsequent block hashes, making tampering detectable.

Database: ~/.prospective/audit/chain.db  (SQLite, no external deps)
"""
from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────── #
# Constants                                                                     #
# ──────────────────────────────────────────────────────────────────────────── #

GENESIS_HASH = "0" * 64   # initial prev_hash for block #1

# Action constants
ACT_LOGIN               = "LOGIN"
ACT_PATIENT_LOADED      = "PATIENT_LOADED"
ACT_SERIES_LOADED       = "SERIES_LOADED"
ACT_SEGMENTATION        = "SEGMENTATION_COMPLETE"
ACT_MESH_EXPORTED       = "MESH_EXPORTED"
ACT_DEVICE_PLACED       = "DEVICE_PLACED"
ACT_DEVICE_REMOVED      = "DEVICE_REMOVED"
ACT_TREATMENT_DECISION  = "TREATMENT_DECISION"
ACT_REPORT_GENERATED    = "REPORT_GENERATED"
ACT_INTEGRITY_CHECK     = "INTEGRITY_CHECK"

_DB_PATH = Path.home() / ".prospective" / "audit" / "chain.db"

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS blocks (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    iso_ts       TEXT    NOT NULL,
    username     TEXT    NOT NULL DEFAULT '',
    patient_hash TEXT    NOT NULL DEFAULT '',
    action       TEXT    NOT NULL,
    payload_json TEXT    NOT NULL DEFAULT '{}',
    payload_hash TEXT    NOT NULL,
    prev_hash    TEXT    NOT NULL,
    block_hash   TEXT    NOT NULL
)
"""


# ──────────────────────────────────────────────────────────────────────────── #
# SkullChain                                                                    #
# ──────────────────────────────────────────────────────────────────────────── #

class SkullChain:
    """Singleton tamper-evident audit trail backed by a local SQLite DB."""

    _instance: Optional["SkullChain"] = None

    # ------------------------------------------------------------------ #
    # Construction / singleton                                             #
    # ------------------------------------------------------------------ #

    def __init__(self) -> None:
        _DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(_DB_PATH), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute(_CREATE_TABLE)
        self._conn.commit()
        # Write genesis block if table is empty
        cur = self._conn.execute("SELECT COUNT(*) FROM blocks")
        if cur.fetchone()[0] == 0:
            self._write_genesis()

    @classmethod
    def instance(cls) -> "SkullChain":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    # ------------------------------------------------------------------ #
    # Internal helpers                                                     #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _compute_block_hash(
        iso_ts: str,
        username: str,
        patient_hash: str,
        action: str,
        payload_hash: str,
        prev_hash: str,
    ) -> str:
        """SHA-256 of all fields joined by '|'."""
        raw = "|".join([iso_ts, username, patient_hash, action, payload_hash, prev_hash])
        return sha256(raw.encode("utf-8")).hexdigest()

    def _last_hash(self) -> str:
        cur = self._conn.execute(
            "SELECT block_hash FROM blocks ORDER BY id DESC LIMIT 1"
        )
        row = cur.fetchone()
        return row[0] if row else GENESIS_HASH

    def _write_genesis(self) -> None:
        """Insert the genesis (sentinel) block so every real block has a prev."""
        iso_ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.") + "000Z"
        action = "GENESIS"
        payload_json = "{}"
        payload_hash = sha256(payload_json.encode()).hexdigest()
        prev_hash = GENESIS_HASH
        block_hash = self._compute_block_hash(
            iso_ts, "", "", action, payload_hash, prev_hash
        )
        self._conn.execute(
            """INSERT INTO blocks
               (iso_ts, username, patient_hash, action,
                payload_json, payload_hash, prev_hash, block_hash)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (iso_ts, "", "", action, payload_json, payload_hash, prev_hash, block_hash),
        )
        self._conn.commit()

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def append(
        self,
        action: str,
        payload: dict,
        username: str = "",
        patient_id: str = "",
        patient_dob: str = "",
    ) -> str:
        """Append a new block to the chain and return its hash."""
        patient_hash = (
            sha256((patient_id + patient_dob).encode("utf-8")).hexdigest()
            if (patient_id or patient_dob)
            else ""
        )
        now = datetime.now(timezone.utc)
        iso_ts = now.strftime("%Y-%m-%dT%H:%M:%S.") + f"{now.microsecond // 1000:03d}Z"
        payload_json = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        payload_hash = sha256(payload_json.encode("utf-8")).hexdigest()
        prev_hash = self._last_hash()
        block_hash = self._compute_block_hash(
            iso_ts, username, patient_hash, action, payload_hash, prev_hash
        )
        self._conn.execute(
            """INSERT INTO blocks
               (iso_ts, username, patient_hash, action,
                payload_json, payload_hash, prev_hash, block_hash)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                iso_ts, username, patient_hash, action,
                payload_json, payload_hash, prev_hash, block_hash,
            ),
        )
        self._conn.commit()
        logger.debug("SkullChain: appended block %s action=%s", block_hash[:12], action)
        return block_hash

    def verify_integrity(self) -> tuple:
        """Verify the entire chain.

        Returns (all_ok: bool, broken: list[dict]) where broken entries are
        dicts with keys: id, iso_ts, action, reason.
        """
        cur = self._conn.execute(
            "SELECT id, iso_ts, username, patient_hash, action, "
            "payload_json, payload_hash, prev_hash, block_hash "
            "FROM blocks ORDER BY id ASC"
        )
        rows = cur.fetchall()
        broken: list[dict] = []
        prev_block_hash: Optional[str] = None

        for row in rows:
            row_id    = row["id"]
            iso_ts    = row["iso_ts"]
            username  = row["username"]
            p_hash    = row["patient_hash"]
            action    = row["action"]
            pl_hash   = row["payload_hash"]
            prev_hash = row["prev_hash"]
            stored_bh = row["block_hash"]

            # Check prev_hash linkage
            if prev_block_hash is None:
                # First block: prev_hash must be GENESIS_HASH
                if prev_hash != GENESIS_HASH:
                    broken.append({
                        "id": row_id,
                        "iso_ts": iso_ts,
                        "action": action,
                        "reason": f"prev_hash mismatch on genesis block (got {prev_hash[:12]}…)",
                    })
            else:
                if prev_hash != prev_block_hash:
                    broken.append({
                        "id": row_id,
                        "iso_ts": iso_ts,
                        "action": action,
                        "reason": (
                            f"prev_hash mismatch: expected {prev_block_hash[:12]}… "
                            f"got {prev_hash[:12]}…"
                        ),
                    })

            # Recompute block hash
            expected_bh = self._compute_block_hash(
                iso_ts, username, p_hash, action, pl_hash, prev_hash
            )
            if expected_bh != stored_bh:
                broken.append({
                    "id": row_id,
                    "iso_ts": iso_ts,
                    "action": action,
                    "reason": (
                        f"block_hash corrupted: expected {expected_bh[:12]}… "
                        f"stored {stored_bh[:12]}…"
                    ),
                })

            prev_block_hash = stored_bh

        all_ok = len(broken) == 0
        return all_ok, broken

    def get_all_blocks(self) -> list:
        """Return all blocks as a list of dicts, ordered by id ASC."""
        cur = self._conn.execute(
            "SELECT id, iso_ts, username, patient_hash, action, "
            "payload_json, payload_hash, prev_hash, block_hash "
            "FROM blocks ORDER BY id ASC"
        )
        rows = cur.fetchall()
        return [dict(r) for r in rows]

    def export_audit_txt(self, path: str) -> None:
        """Write a human-readable text audit report to *path*."""
        blocks = self.get_all_blocks()
        all_ok, broken = self.verify_integrity()

        col_widths = [5, 28, 16, 30, 16, 16]
        headers = ["#", "Fecha/Hora", "Usuario", "Acción", "Paciente(hash)", "Hash bloque"]

        def _row(vals: list) -> str:
            return " | ".join(
                str(v).ljust(w)[:w] for v, w in zip(vals, col_widths)
            )

        sep = "-+-".join("-" * w for w in col_widths)

        lines: list[str] = [
            "=" * 120,
            "  PROSPECTIVE — SkullChain™ Audit Trail Export",
            f"  Generated: {datetime.now(timezone.utc).isoformat(timespec='seconds')}Z",
            "=" * 120,
            "",
            _row(headers),
            sep,
        ]

        for blk in blocks:
            pat_disp = (blk["patient_hash"][:8] + "…") if blk["patient_hash"] else "—"
            bh_disp  = blk["block_hash"][:12] + "…"
            lines.append(_row([
                blk["id"],
                blk["iso_ts"],
                blk["username"] or "—",
                blk["action"],
                pat_disp,
                bh_disp,
            ]))

        lines += [
            sep,
            "",
            f"Total blocks: {len(blocks)}",
        ]

        if all_ok:
            lines.append(f"Integrity: OK — all {len(blocks)} blocks verified.")
        else:
            lines.append(f"Integrity: WARNING — {len(broken)} corrupted block(s) detected!")
            for b in broken:
                lines.append(f"  Block #{b['id']} ({b['iso_ts']}) {b['action']}: {b['reason']}")

        lines.append("=" * 120)

        with open(path, "w", encoding="utf-8") as fh:
            fh.write("\n".join(lines) + "\n")
