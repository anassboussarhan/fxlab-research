"""Freeze the final candidate configurations (Phase 11). Writes results/frozen_candidates.json containing the
exact candidate configs, the code commit, a hash of the development results database and the SHA-256 of an
unlock passphrase chosen by the human authoriser. The 2026 holdout loaders stay locked until
FXLAB_HOLDOUT_UNLOCK=<passphrase> is set AND this file exists (explicit authorisation)."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fxlab.config import FROZEN_CANDIDATES_FILE, RESULTS_DIR  # noqa: E402
from fxlab.orchestration.campaign import git_commit, DB_PATH  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", required=True, help="JSON file with the list of frozen candidate configs")
    ap.add_argument("--passphrase-sha256", required=True, help="sha256 of the authoriser's unlock passphrase")
    a = ap.parse_args()
    if FROZEN_CANDIDATES_FILE.exists():
        sys.exit("candidates already frozen; refusing to overwrite (development decisions are locked)")
    cands = json.load(open(a.candidates))
    db_hash = hashlib.sha256(open(DB_PATH, "rb").read()).hexdigest() if DB_PATH.exists() else None
    FROZEN_CANDIDATES_FILE.write_text(json.dumps({
        "frozen": True, "frozen_at": datetime.now(timezone.utc).isoformat(), "code_commit": git_commit(),
        "results_db_sha256": db_hash, "candidates": cands, "unlock_token_sha256": a.passphrase_sha256,
        "note": "Evaluate 2026 once. If evaluated before year end, report as YEAR-TO-DATE holdout."}, indent=1))
    print("frozen", len(cands), "candidates ->", FROZEN_CANDIDATES_FILE)


if __name__ == "__main__":
    main()
