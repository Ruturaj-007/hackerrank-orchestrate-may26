from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


class RunLogger:
    """Logs each ticket decision to log.txt and stdout."""

    def __init__(self, log_path: str | Path = "log.txt"):
        self._path   = Path(log_path)
        self._start  = time.time()
        self._count  = 0
        self._errors = 0
        self._fh     = self._path.open("w", encoding="utf-8")
        self._write_header()

    # internal 

    def _write_header(self):
        self._fh.write(
            json.dumps({
                "event": "run_start",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }) + "\n"
        )
        self._fh.flush()

    def _write(self, record: dict):
        self._fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._fh.flush()

    # public API 

    def log_ticket(
        self,
        row_idx:       int,
        issue:         str,
        subject:       str,
        company:       Optional[str],
        status:        str,
        product_area:  str,
        request_type:  str,
        response:      str,
        justification: str,
        retrieval_sources: list[str],
        escalation_rule:   str,
        elapsed_s:         float,
    ):
        self._count += 1
        record = {
            "event":             "ticket_processed",
            "row":               row_idx,
            "company":           company,
            "subject":           subject[:80] if subject else "",
            "issue_preview":     issue[:120] if issue else "",
            "status":            status,
            "product_area":      product_area,
            "request_type":      request_type,
            "response_preview":  response[:120] if response else "",
            "justification":     justification,
            "retrieval_sources": retrieval_sources,
            "escalation_rule":   escalation_rule,
            "elapsed_s":         round(elapsed_s, 2),
        }
        self._write(record)
        self._print_progress(row_idx, status, product_area, elapsed_s)

    def log_error(self, row_idx: int, error: str):
        self._errors += 1
        self._write({"event": "error", "row": row_idx, "error": error})
        print(f"  [ERROR] row {row_idx}: {error}", file=sys.stderr)

    def close(self, total_rows: int):
        elapsed = time.time() - self._start
        self._write({
            "event":        "run_complete",
            "timestamp":    datetime.now(timezone.utc).isoformat(),
            "total_rows":   total_rows,
            "processed":    self._count,
            "errors":       self._errors,
            "elapsed_s":    round(elapsed, 2),
        })
        self._fh.close()
        print(
            f"\n{'─'*60}\n"
            f"  Done. {self._count}/{total_rows} tickets processed "
            f"({self._errors} errors) in {elapsed:.1f}s\n"
            f"  Log: {self._path}\n"
            f"{'─'*60}"
        )

    # terminal output 

    def _print_progress(self, row_idx: int, status: str, product_area: str, elapsed: float):
        icon   = "✓" if status == "replied" else "⚠"
        status_display = status.upper()
        print(
            f"  [{icon}] row {row_idx:>3} | {status_display:<10} | "
            f"{product_area:<30} | {elapsed:.1f}s"
        )