from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from analysis.extra import ExtraAnalysis
from analysis.pairs import PairAnalysis
from analysis.records import load_records
from analysis.signals import SignalAnalysis


def _write(out: Path, name: str, result: dict[str, Any]) -> None:
    path = out / f"{name}.json"
    path.write_text(json.dumps(result, indent=1, default=float))
    print(f"wrote {path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize offline measurement JSONs into tables."
    )
    parser.add_argument("--signals", type=Path, required=True)
    parser.add_argument("--extra", type=Path)
    parser.add_argument("--pairs", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    records = load_records(args.signals)
    signals = SignalAnalysis(records).run()
    _write(args.out, "signals", signals)
    print(json.dumps(signals["verdicts"], indent=1, default=float))
    if args.extra is not None:
        _write(args.out, "extra", ExtraAnalysis(records, args.extra).run())
    if args.pairs is not None:
        pairs = PairAnalysis(load_records(args.pairs), args.signals)
        _write(args.out, "pairs", pairs.run())


if __name__ == "__main__":
    main()
