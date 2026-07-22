#!/usr/bin/env python3
"""
report.py - turn results.csv into a wide, paste-into-Google-Sheets table
            that puts cuD-PDLP (cuopt-distributed) head-to-head with D-PDLP.

Output is TSV (tab-separated). 3 header rows + one row per instance.

Two areas:

  1. DATA BLOCKS -- raw values per (solver, N).
       header row layout:  row1 = metric ("step (s)", "peak (GB)", ...),
                           row2 = solver, row3 = GPU count
       columns:            step (s), peak (GB), peak2 (GB)
                           for cuD-PDLP, D-PDLP, cuopt-base.

  2. COMPARISON AREA -- compact "by GPU count" head-to-head.
       header row layout:  row1 = GPU count ("1 GPU", "2 GPUs", ...),
                           row2 = sub-metric ("speedup", "peak (GB)",
                                              "peak2 (GB)"),
                           row3 = reference / solver
       per N:              [speedup vs cuD-PDLP N=1] [speedup vs D-PDLP]
                           [peak    cuD-PDLP]        [peak    D-PDLP]
                           [peak2   cuD-PDLP]        [peak2   D-PDLP]
       Both speedup cells describe cuD-PDLP-at-N; the reference differs:
         vs cuD-PDLP N=1 = T_cuopt(1) / T_cuopt(N)   (within-cuopt scaling)
         vs D-PDLP       = T_dpdlp(N) / T_cuopt(N)   (head-to-head at this N)
       In both, a value > 1 means cuD-PDLP is faster than the reference.
       Peak / peak2 cells just reference the data block via INDIRECT.

Memory units: peak / peak2 are reported in GiB (binary, value ÷ 1024
from the MB stored in results.csv). This matches the "GiB-on-the-card"
numbers you'd compare against device limits (e.g. ~180 GiB on a B200).

Locale note:
  Default formulas use `,` as the function argument separator
  (US-locale Google Sheets). If your sheet's locale is FR / DE / IT /
  ... it likely uses `;` instead; pass `--sep ';'` in that case.

Usage:
  python report.py                       # write report.tsv (US separator)
  python report.py --sep ';'             # EU-locale formulas
  python report.py -o out.tsv            # custom output
  python report.py --csv path/to/x.csv   # custom input CSV
  python report.py --stdout              # also print the TSV to stdout
  python report.py --n-list 1,2,4,8      # restrict the GPU counts shown

Stdlib only.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

# Pretty solver labels for the spreadsheet headers. Internal CSV names are
# 'cuopt-distributed', 'dpdlp', 'cuopt-base'; we show them as the names
# the user actually uses when talking about the work.
SOLVER_LABEL = {
    "cuopt-distributed": "cuD-PDLP",
    "dpdlp":             "D-PDLP",
    "cuopt-base":        "cuopt-base",
}

# Order matters: column groups appear left-to-right in this order.
SOLVER_ORDER = ["cuopt-distributed", "dpdlp", "cuopt-base"]


def col_letter(idx0: int) -> str:
    """Convert a 0-based column index to a spreadsheet letter (A, B, ..., Z, AA, AB, ...)."""
    s, n = "", idx0 + 1
    while n > 0:
        n, r = divmod(n - 1, 26)
        s = chr(ord("A") + r) + s
    return s


def parse_cli() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    here = Path(__file__).resolve().parent
    p.add_argument("--csv", type=Path, default=here / "results.csv",
                   help="path to results.csv. Default: %(default)s")
    p.add_argument("-o", "--output", type=Path, default=here / "report.tsv",
                   help="path to the output TSV. Default: %(default)s")
    p.add_argument("--n-list", default="1,2,4,8",
                   help="comma-separated list of GPU counts to include for the "
                        "distributed solvers. Default: %(default)s")
    p.add_argument("--sep", default=",", choices=(",", ";"),
                   help="formula argument separator. US-locale sheets use ','; "
                        "Google Sheets in EU / FR locales uses ';'. "
                        "Default: %(default)s")
    p.add_argument("--stdout", action="store_true",
                   help="also print the TSV to stdout")
    return p.parse_args()


def _to_float(s: str) -> float | None:
    """Parse a CSV cell into a float. Empty / unparseable -> None."""
    if s is None:
        return None
    s = s.strip()
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def load_rows(csv_path: Path) -> dict[tuple[str, str, int], dict]:
    """Index results.csv by (solver, instance, n_gpus)."""
    if not csv_path.is_file():
        sys.exit(f"results CSV not found: {csv_path}")
    out: dict[tuple[str, str, int], dict] = {}
    with csv_path.open() as f:
        for row in csv.DictReader(f):
            try:
                key = (row["solver"], row["instance"], int(row["n_gpus"]))
            except (KeyError, ValueError):
                continue
            out[key] = {
                "step_s":       _to_float(row.get("step_s", "")),
                "total_s":      _to_float(row.get("total_s", "")),
                "gpu_peak_mb":  _to_float(row.get("gpu_peak_mb", "")),
                "gpu_peak2_mb": _to_float(row.get("gpu_peak2_mb", "")),
                "iterations":   row.get("iterations", ""),
                "status":       row.get("status", ""),
            }
    return out


def instance_order(rows: dict, csv_path: Path) -> list[str]:
    """Order instances by bench.INSTANCES if importable, else alphabetically.

    Anything in the CSV but not in bench.INSTANCES (e.g. retired entries)
    goes at the end, alphabetically.
    """
    in_csv = sorted({inst for _, inst, _ in rows.keys()})
    try:
        sys.path.insert(0, str(csv_path.parent))
        import bench  # type: ignore[import]
        wanted = [Path(p).name for p in bench.INSTANCES]
        ordered = [i for i in wanted if i in in_csv]
        extras = [i for i in in_csv if i not in wanted]
        return ordered + extras
    except Exception:
        return in_csv


def fmt_num(x: float | None) -> str:
    if x is None:
        return ""
    if x == 0:
        return "0"
    # Use fixed notation for typical values; fall back to scientific for very small/large.
    if 1e-3 <= abs(x) < 1e7:
        return f"{x:.6g}"
    return f"{x:.6e}"


def _ngpu_label(n: int) -> str:
    return "1 GPU" if n == 1 else f"{n} GPUs"


def build_table(rows: dict, n_list: list[int], sep: str = ",",
                ) -> tuple[list[list[str]], list[list[str]]]:
    """Build the wide TSV (3-row header + data rows).

    Each column is associated with a (metric, solver, n_gpus) triple.
    The three header rows are derived from those triples:
        row 1 (metric_row) : "step (s)", "peak (MB)", "speedup vs N=1", ...
        row 2 (solver_row) : "cuD-PDLP", "D-PDLP", "cuopt-base" (or "")
        row 3 (ngpu_row)   : "1 GPU", "2 GPUs", "4 GPUs", "8 GPUs"
    A row-N label only appears in the first column where (metric, solver)
    *changes*; subsequent columns in the same block leave it empty so the
    user can visually merge them in Sheets if desired.

    Data starts at row 4 (1-based) in the resulting spreadsheet.

    Formulas reference cells via INDIRECT("<col>" & ROW()) so they pick
    up cells in the SAME spreadsheet row they're pasted into. Columns
    are hard-coded by letter, so a formula is portable across any paste
    target (A1, A4, A47, ...).

    Returns:
        header_rows : list of 3 lists [metric_row, solver_row, ngpu_row]
        out_rows    : one cell list per instance (same width as headers)

    `sep` controls the spreadsheet function-arg separator (`,` for US
    locales, `;` for EU / FR locales).
    """
    instances = instance_order(rows, Path("results.csv"))

    # ----- which (solver, N) pairs exist in this dataset -----
    pairs: list[tuple[str, int]] = []
    for s in SOLVER_ORDER:
        ns = n_list if s != "cuopt-base" else [1]  # cuopt-base only sweeps N=1
        for n in ns:
            pairs.append((s, n))

    # The three header rows + the data column metadata. We track
    # (metric, solver) per column to know when to emit a non-empty
    # label vs an empty cell (for the block-start "merge effect").
    metric_row: list[str] = [""]
    solver_row: list[str] = [""]
    ngpu_row:   list[str] = ["instance"]
    col_meta:   list[tuple[str, str]] = [("", "")]  # (metric, solver)

    def _add_col(metric: str, solver_label: str, n: int) -> int:
        """Append one column; return its 0-based index. Row-1 / row-2
        labels are emitted only when (metric) / (metric, solver)
        transitions occur, so consecutive same-block columns are blank."""
        idx = len(ngpu_row)
        # metric_row: non-empty only at the very first column of each metric block.
        prev_metric = col_meta[-1][0] if col_meta else ""
        prev_solver = col_meta[-1][1] if col_meta else ""
        metric_row.append(metric if metric != prev_metric else "")
        # solver_row: non-empty when (metric, solver) changes.
        new_block = (metric != prev_metric) or (solver_label != prev_solver)
        solver_row.append(solver_label if new_block else "")
        ngpu_row.append(_ngpu_label(n))
        col_meta.append((metric, solver_label))
        return idx

    # ----- DATA BLOCKS: step / peak / peak2 -----
    # Memory is reported in GiB (binary, ÷1024) rather than MB.
    # NVML's underlying `used` field is bytes, bench.py divides by 1024*1024
    # to get the MB stored in the CSV, so dividing by 1024 here gives the
    # exact "GiB on the card" number you'd compare to the device limit
    # (e.g. ~180 GiB on a B200).
    MB_PER_GB = 1024.0

    def _mb_to_gb(v: float | None) -> float | None:
        return None if v is None else v / MB_PER_GB

    step_idx:  dict[tuple[str, int], int] = {}
    peak_idx:  dict[tuple[str, int], int] = {}
    peak2_idx: dict[tuple[str, int], int] = {}

    for s, n in pairs:
        step_idx[(s, n)] = _add_col("step (s)", SOLVER_LABEL[s], n)
    for s, n in pairs:
        peak_idx[(s, n)] = _add_col("peak (GB)", SOLVER_LABEL[s], n)
    for s, n in pairs:
        peak2_idx[(s, n)] = _add_col("peak2 (GB)", SOLVER_LABEL[s], n)

    # ----- emit the data rows -----
    out_rows: list[list[str]] = []
    for inst in instances:
        row_cells: list[str] = [inst] + [""] * (len(ngpu_row) - 1)
        for s, n in pairs:
            d = rows.get((s, inst, n))
            if d is None:
                continue
            row_cells[step_idx[(s, n)]]  = fmt_num(d.get("step_s"))
            row_cells[peak_idx[(s, n)]]  = fmt_num(_mb_to_gb(d.get("gpu_peak_mb")))
            row_cells[peak2_idx[(s, n)]] = fmt_num(_mb_to_gb(d.get("gpu_peak2_mb")))
        out_rows.append(row_cells)

    # ----- COMPARISON AREA (by GPU count) -----
    # Three header rows in THIS area carry different meanings than in the
    # data blocks above:
    #     row 1 = GPU count          ("1 GPU", "2 GPUs", "4 GPUs", "8 GPUs")
    #     row 2 = sub-metric         ("speedup", "peak (GB)", "peak2 (GB)")
    #     row 3 = solver             ("cuD-PDLP", "D-PDLP")
    # Layout per N (6 cells):
    #     [speedup cuD-PDLP] [speedup D-PDLP] [peak cuD-PDLP] [peak D-PDLP] [peak2 cuD-PDLP] [peak2 D-PDLP]
    # speedup at N=1 is T(1)/T(1) = 1.0 by construction; kept for a
    # uniform table shape. peak / peak2 cells are direct INDIRECT
    # references to the corresponding data-block cell (one cell deep).
    def _add_col_raw(top: str, mid: str, bot: str) -> int:
        """Append a column with explicit header text for each row.
        Caller decides when to leave a row blank to get the 'merge
        effect' for block-spanning labels."""
        metric_row.append(top)
        solver_row.append(mid)
        ngpu_row.append(bot)
        col_meta.append((top, mid))
        return len(ngpu_row) - 1

    SOLVERS_IN_COMP = ("cuopt-distributed", "dpdlp")
    cmp_solver_label = {
        "cuopt-distributed": SOLVER_LABEL["cuopt-distributed"],  # "cuD-PDLP"
        "dpdlp":             SOLVER_LABEL["dpdlp"],              # "D-PDLP"
    }

    def _emit_cell(top: str, mid: str, bot: str, formula: str) -> None:
        _add_col_raw(top, mid, bot)
        for row_cells in out_rows:
            row_cells.append(formula)

    def _ref(idx: int) -> str:
        """Return an INDIRECT reference to (col, current row), no IFERROR.
        Used for direct lookups of an existing data cell."""
        return f'=INDIRECT("{col_letter(idx)}"&ROW())'

    def _ratio(num_idx: int, den_idx: int) -> str:
        """Return =IFERROR(INDIRECT(num,row)/INDIRECT(den,row), "")."""
        n = col_letter(num_idx)
        d = col_letter(den_idx)
        return (f'=IFERROR(INDIRECT("{n}"&ROW())'
                f'/INDIRECT("{d}"&ROW()){sep}"")')

    for n in n_list:
        ngpu_top = _ngpu_label(n)

        # ----- speedup sub-block -----
        # Both cells describe cuD-PDLP-at-N (the subject); the *reference*
        # differs:
        #   cell 1: vs cuD-PDLP at 1 GPU  -> T_cuopt(1) / T_cuopt(N)
        #             within-cuopt scaling; at N=1 this is trivially 1.0.
        #   cell 2: vs D-PDLP at the same N -> T_dpdlp(N) / T_cuopt(N)
        #             cross-solver head-to-head at this N.
        # In both, a value > 1 means cuD-PDLP wins.
        cuopt_ref_n1 = step_idx.get(("cuopt-distributed", 1))
        cuopt_ref_n  = step_idx.get(("cuopt-distributed", n))
        dpdlp_ref_n  = step_idx.get(("dpdlp", n))

        # cell 1: vs cuD-PDLP N=1
        if cuopt_ref_n1 is not None and cuopt_ref_n is not None:
            formula = _ratio(num_idx=cuopt_ref_n1, den_idx=cuopt_ref_n)
        else:
            formula = ""
        _emit_cell(
            top=ngpu_top,
            mid="speedup",
            bot="vs cuD-PDLP N=1",
            formula=formula,
        )

        # cell 2: vs D-PDLP at the same N
        if dpdlp_ref_n is not None and cuopt_ref_n is not None:
            formula = _ratio(num_idx=dpdlp_ref_n, den_idx=cuopt_ref_n)
        else:
            formula = ""
        _emit_cell(
            top="",
            mid="",
            bot="vs D-PDLP",
            formula=formula,
        )

        # ----- peak (GB) sub-block (direct data-block reference) -----
        for k, s in enumerate(SOLVERS_IN_COMP):
            formula = _ref(peak_idx[(s, n)]) if (s, n) in peak_idx else ""
            _emit_cell(
                top="",
                mid="peak (GB)" if k == 0 else "",
                bot=cmp_solver_label[s],
                formula=formula,
            )

        # ----- peak2 (GB) sub-block (direct data-block reference) -----
        for k, s in enumerate(SOLVERS_IN_COMP):
            formula = _ref(peak2_idx[(s, n)]) if (s, n) in peak2_idx else ""
            _emit_cell(
                top="",
                mid="peak2 (GB)" if k == 0 else "",
                bot=cmp_solver_label[s],
                formula=formula,
            )

    header_rows = [metric_row, solver_row, ngpu_row]
    return header_rows, out_rows


def main() -> int:
    args = parse_cli()
    n_list = [int(x) for x in args.n_list.split(",") if x.strip()]

    rows = load_rows(args.csv)
    if not rows:
        sys.exit(f"results CSV is empty: {args.csv}")

    header_rows, out_rows = build_table(rows, n_list, sep=args.sep)
    metric_row, solver_row, ngpu_row = header_rows

    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Manual TSV writer (not csv.writer): the formula cells contain literal
    # double quotes (e.g. `=IFERROR(...,"")`) which csv.writer would
    # quote-escape into `"=IFERROR(...,"""")"`. Google Sheets paste-special
    # interprets TSV without unescaping CSV-style quotes, so it would land
    # the literal string in the cell instead of activating the formula.
    # No cell ever contains a tab or newline by construction.
    def _tsv_cell(x: str) -> str:
        # Defensive: strip stray tabs/newlines just in case a future
        # parsed value ever sneaks one in (e.g. an LP file name).
        return x.replace("\t", " ").replace("\n", " ").replace("\r", " ")

    with args.output.open("w") as f:
        # Row 1: metric group  (e.g. "step (s)", "speedup vs N=1", ...).
        # Row 2: solver        (e.g. "cuD-PDLP", "D-PDLP", "cuopt-base").
        # Row 3: GPU count     (e.g. "1 GPU", "2 GPUs", ..., or "instance").
        # Rows 4+: per-instance data + INDIRECT-based formulas. Each
        # block's first column carries its row-1/row-2 label; the rest
        # are blank so you can visually merge them in Sheets if you want.
        # Skip whichever header rows you don't need when copying.
        for row in (metric_row, solver_row, ngpu_row):
            f.write("\t".join(_tsv_cell(c) for c in row) + "\n")
        for row_cells in out_rows:
            f.write("\t".join(_tsv_cell(c) for c in row_cells) + "\n")

    print(
        f"report: {len(out_rows)} instance row(s), {len(ngpu_row)} columns "
        f"total -> {args.output}\n"
        f"        3 header rows: row 1 = metric, row 2 = solver, "
        f"row 3 = GPU count\n"
        f"        formulas use INDIRECT(\"<col>\"&ROW()) so paste anywhere",
        file=sys.stderr,
    )

    if args.stdout:
        with args.output.open() as f:
            sys.stdout.write(f.read())

    return 0


if __name__ == "__main__":
    sys.exit(main())
