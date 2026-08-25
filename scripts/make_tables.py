"""Regenerate manuscript tables from stored result files.

Usage
-----
    python scripts/make_tables.py --table 5 --results-dir results
    python scripts/make_tables.py --table all --format markdown \
        --out results/tables

Hard rules enforced here
------------------------
1. Numbers come only from ``results/`` files that an actual run produced. There
   is no fallback path that prints manuscript values.
2. A table whose inputs are missing prints the commands that would produce them
   and exits non-zero. It never prints a partially fabricated table.
3. A table whose manuscript definition is internally inconsistent is not
   emitted; the audit finding is printed instead.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from vrfraudnet.tables.build_tables import (  # noqa: E402
    TableBuildError,
    render_markdown,
    table_5_primary_performance,
    table_7_ablation,
    table_8_transfer,
    table_9_temporal,
    table_10_evasion,
)

log = logging.getLogger("make_tables")

TABLE_BUILDERS = {
    "5": table_5_primary_performance,
    "7": table_7_ablation,
    "8": table_8_transfer,
    "9": table_9_temporal,
    "10": table_10_evasion,
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table", default="all",
                        choices=("all", *sorted(TABLE_BUILDERS)))
    parser.add_argument("--results-dir", type=Path, default=REPO_ROOT / "results")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "results" / "tables")
    parser.add_argument("--format", default="markdown", choices=("markdown",))
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level), format="%(levelname)s | %(message)s")
    args.out.mkdir(parents=True, exist_ok=True)

    wanted = sorted(TABLE_BUILDERS) if args.table == "all" else [args.table]
    failures = 0
    for name in wanted:
        try:
            rendered = TABLE_BUILDERS[name](args.results_dir)
        except TableBuildError as exc:
            failures += 1
            log.error("Table %s could not be built:\n%s", name, exc)
            continue
        path = args.out / f"table_{name}.md"
        path.write_text(render_markdown(rendered), encoding="utf-8")
        log.info("wrote %s", path)

    if failures:
        log.error(
            "%d table(s) could not be built. This repository never substitutes "
            "manuscript values for missing runs.", failures,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
