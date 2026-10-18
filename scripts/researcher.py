#!/usr/bin/env python3
"""Researcher CLI — trackable progress/inventory for the JOIV revision.

Usage:
  python scripts/researcher.py status    # manifest summary + live gpu-kc state
  python scripts/researcher.py check     # validate manifest files/evidence exist
  python scripts/researcher.py refresh   # stamp manifest with today's date + git HEAD
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "RESEARCH_MANIFEST.yaml"


def _read() -> str:
    return MANIFEST.read_text()


def status() -> int:
    print(_read())
    try:
        out = subprocess.run(
            ["ssh", "-o", "ConnectTimeout=15", "-o", "BatchMode=yes", "gpu-kc",
             'echo "SWEEP_DONE=$(grep -c \\"SWEEP DONE\\" /mnt/home/dzakirm/histodae_joiv/sweep.log)"; '
             'echo "SCOUT_DONE=$(grep -c \\"SCOUT20 DONE\\" /mnt/home/dzakirm/histodae_joiv/scout20.log)"; '
             'ls /tmp/joiv_gaps/kmag 2>/dev/null | wc -l'],
            capture_output=True, text=True, timeout=60)
        print("--- live gpu-kc ---\n" + (out.stdout or out.stderr))
    except Exception as e:  # noqa: BLE001
        print(f"--- live gpu-kc ---\nunreachable: {e}")
    return 0


def check() -> int:
    import re
    text = _read()
    bad = 0
    for rel in re.findall(r"(joiv_revision/[\w./-]+|paper_rev/[\w. ()-]+|RESEARCH_MANIFEST\.yaml)", text):
        p = ROOT / rel
        ok = p.exists()
        print(("OK  " if ok else "MISS"), rel)
        bad += (not ok)
    for doc in ["joiv_revision/LATEST_REVISION.md",
                "joiv_revision/response/response_letter_draft.md",
                "joiv_revision/results/frozen_metrics.csv",
                "joiv_revision/manuscript/JOIV_revised_v1.docx",
                "DOCUMENTATION_INDEX.md"]:
        ok = (ROOT / doc).exists()
        print(("OK  " if ok else "MISS"), doc)
        bad += (not ok)
    print(f"\ncheck: {'PASS' if bad == 0 else f'{bad} missing'}")
    return 1 if bad else 0


def refresh() -> int:
    import datetime
    head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                          capture_output=True, text=True).stdout.strip()
    stamp = f"# refreshed: {datetime.date.today().isoformat()} | git: {head}\n"
    text = _read()
    text = stamp + "\n".join(l for l in text.splitlines() if not l.startswith("# refreshed:"))
    MANIFEST.write_text(text + "\n")
    print(stamp.strip())
    return 0


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "status"
    sys.exit({"status": status, "check": check, "refresh": refresh}.get(cmd, status)())
