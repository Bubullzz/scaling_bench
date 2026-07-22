# scaling_bench

Strong / weak scaling sweep for Distributed PDLP (`bullshit_mostovoi_cuopt`).

## What it measures

For every `(problem, n_gpus)` pair the sweep runs `cuopt_cli` with:

- `use_distributed_pdlp = true`, `method = 1` (PDLP-only), `presolve = 0` (None),
  `pdlp_precision = 1` (double)
- `iteration_limit = 5000`, all six tolerances pinned to `1e-30`

so each invocation does **fixed work** (5000 PDLP iterations) and is directly
comparable across `n_gpus`. Per-major-iteration table rows plus the patched
binary's explicit `Setup time:` / `Step time:` log lines give a setup-free
"time per iteration" that is the cleanest scaling metric.

## Layout

```
scaling_bench/
  config/
    distributed_pdlp.params       # fixed cuopt_cli params (5000 iters, tight tol)
  problems.list                   # one absolute MPS path per line
  gpu_counts.list                 # one GPU count per line (1, 2, 4, ...)
  weak_scaling_groups.yaml.example # template for weak-scaling families
  partitions/                     # optional cached METIS partitions
  runs/                           # one log per run after run_sweep.sh
  results.csv                     # one row per timed run (after aggregate.py)
  results_summary.csv             # one row per (problem, N) with median/CoV
  plots/                          # PNGs from plot_scaling.py
  scripts/
    run_sweep.sh                  # main driver
    parse_log.py                  # stdlib log parser
    aggregate.py                  # builds the two CSVs
    plot_scaling.py               # generates plots (needs matplotlib)
    generate_partitions.sh        # optional: cache METIS partitions
```

## Prerequisites

- `bullshit_mostovoi_cuopt` built with the run_solver patch. The binary
  at `bullshit_mostovoi_cuopt/cpp/build/cuopt_cli` is the default; you
  can override with `--binary`.
- A GPU host with N GPUs visible via `nvidia-smi -L`.
- `python3` (stdlib) for `parse_log.py` and `aggregate.py`.
- For plots: `matplotlib`, `pandas`, `numpy`. The `d_pdlp_env` conda env
  on this machine has them at:
  `/home/scratch.vmostovoi_gpu/.conda/envs/d_pdlp_env/bin/python3`.

## Runbook

### 1. Edit `problems.list` and `gpu_counts.list`

Add the MPS files (absolute paths) and GPU counts (powers of 2 up to your
visible count).

```bash
cd /home/scratch.vmostovoi_gpu/scaling_bench
$EDITOR problems.list gpu_counts.list
```

### 2. (Optional) Pre-cache METIS partitions

Reduces rep-to-rep variance. Skip if you don't care.

```bash
bash scripts/generate_partitions.sh
```

### 3. Dry-run the sweep

Validates input files and prints what would run without invoking
`cuopt_cli`. Especially useful on a non-GPU host.

```bash
bash scripts/run_sweep.sh --dry-run
```

### 4. Execute the sweep

```bash
bash scripts/run_sweep.sh --warmups 1 --reps 3
```

Logs land in `runs/<stem>__N<n>__rep<id>.log`. Warm-ups carry `rep=W0,
W1, ...` and are excluded from the summary CSV. Resume is automatic:
existing non-empty timed logs are skipped unless `--force` is passed.

### 5. Aggregate

```bash
python3 scripts/aggregate.py
```

Writes `results.csv` (one row per timed run; warm-ups still included
with `is_warmup=True`) and `results_summary.csv` (one row per
`(problem, n_gpus)` with median / mean / stdev / CoV of the timing
metrics plus an `iters_consistent` flag).

The five timing-related columns mean:

| Column | What it is | Use it for |
| --- | --- | --- |
| `setup_time_s` | Time before the PDLP loop starts (solver ctor, METIS partition, NCCL bootstrap, distributed Ruiz + Pock-Chambolle scaling, sigma_max power iter). On a patched cuopt this comes from the binary's explicit `Setup time:` log line; on an unpatched binary it's derived from the elapsed value on the first per-major-iter row. | Reporting setup cost / Amdahl serial fraction |
| `setup_time_source` | `"explicit"` (patched binary, authoritative) or `"first_iter_row"` (fallback). | Provenance |
| **`step_time_s`** | **Time spent inside the PDLP iteration loop only. THIS is the scaling metric.** On a patched cuopt this comes from the explicit `Step time:` log line; otherwise it's `total_time_s − setup_time_s`. | **Strong / weak scaling** |
| **`time_per_step_ms`** | `step_time_s × 1000 / iterations`. Steady-state per-step cost. | **Per-iteration scaling curve** |
| `total_time_s` | Full wall-clock from the `Status: ... Time: X.Xs` line. | Reporting total run time only |
| `loop_time_s`, `time_per_iter_ms` | Secondary: delta between the first and last rows of the per-major-iter table. Robust if the first row didn't print at iter 0. | Sanity-check against `step_time_s` |

### 6. Sanity-check (do this BEFORE trusting the numbers)

Open `results_summary.csv` and verify:

- `iters_consistent == True` for every row (== all reps did exactly the
  same number of PDLP iterations; if not, the fixed-work assumption is
  broken — usually due to numerical issues at high N, see Troubleshooting).
- `status_consensus == "Iteration Limit"` everywhere (we tightened
  tolerances specifically so every run hits this).
- `time_per_step_ms_cov` ideally ≤ 5%. Higher than that → noisy machine
  or partition caching is needed; bump `--reps`.
- `parse_warnings_any == False` everywhere; if not, inspect the
  corresponding `runs/*.log` and look at `parse_warnings` in `results.csv`.
- `setup_fraction_median` should grow with N (METIS/NCCL/scaling get
  heavier with more shards). If it's ≥ ~0.3 at your largest N, you'll see
  a much bigger gap between speedup-on-`step_time_s` (the truth) and
  speedup-on-`total_time_s` (setup-contaminated). The `step_time_s`
  speedup is the one to report.

### 7. Plot

```bash
/home/scratch.vmostovoi_gpu/.conda/envs/d_pdlp_env/bin/python3 \
  scripts/plot_scaling.py
```

Outputs (all based on **step time**, i.e. setup excluded — that's what we
care about for scaling):

- `plots/strong/<stem>_speedup.png` — log-log speedup vs N with ideal y=x.
- `plots/strong/<stem>_time_per_step.png` — per-step time vs N.
- `plots/strong/<stem>_efficiency.png` — `speedup / N` vs N (linear).
- `plots/strong/all_speedup.png` — all problems overlaid.

For weak scaling, copy `weak_scaling_groups.yaml.example` to
`weak_scaling_groups.yaml`, fill in the (stem, N) pairs that should give a
flat curve, and re-run `plot_scaling.py`. The plot lives at
`plots/weak/<group_name>_time_per_step.png`.

## Computing the scaling numbers by hand

If you want to do your own analysis from `results.csv`:

- **Strong scaling, problem p**: for each N,
  `speedup(p, N) = median(step_time_s | p, N=1) / median(step_time_s | p, N)`.
  Equivalently `time_per_step_ms(p, 1) / time_per_step_ms(p, N)`. Efficiency
  is `speedup / N`. *Do not use `total_time_s` for this* — setup grows with
  N and will undersell the speedup, sometimes substantially.
- **Weak scaling, family `(p_k, N_k)`**: just plot `time_per_step_ms(p_k, N_k)`
  vs `N_k`. Should be flat under ideal weak scaling.
- **Setup overhead** (already in summary as `setup_fraction_median`):
  `setup_time_s / total_time_s` is the Amdahl serial fraction proxy.

## Binary-side support for step-only timing

`bullshit_mostovoi_cuopt` carries a patch in `cpp/src/pdlp/{pdlp.cuh,
pdlp.cu, solve.cu}` that adds two new log lines to the distributed
solver output:

```
Setup time: 1.823s (before PDLP iteration loop)
...
Status: Iteration Limit   Objective: +1.0e+00  Iterations: 5000  Time: 23.812s
Step time: 21.989s   Time/step: 4.398ms   (setup 1.823s of 23.812s total, 7.7%)
```

The first line is emitted inside `pdlp_solver_t::run_solver` right
before the main `while (true)` loop and is governed by a new member
`setup_time_s_` (exposed via `get_setup_time_s()`). The second is
emitted by `solve_lp_distributed_from_mps` immediately after the
existing `Status:` line, using that getter.

The parser auto-detects these lines and sets `setup_time_source =
"explicit"` when present; otherwise it falls back to deriving setup
from the first iter-table row. Either way the downstream
`step_time_s` / `time_per_step_ms` columns and plots are correct.

## Troubleshooting

| Symptom | Likely cause | Fix |
| --- | --- | --- |
| `iters_consistent=False` for some `(problem, N)` | Numerical drift makes PDLP early-exit on some reps. Distributed scaling for very large problems can hit `Status: A numerical error...` | Verify tolerances in `config/distributed_pdlp.params` are at `1e-30`. If still flaky, raise `iteration_limit` and bump `--reps`. |
| `time_per_step_ms_cov > 10%` for a given `(problem, N)` | METIS produced different partitions across reps, or another tenant is on the box | Run `scripts/generate_partitions.sh` (Step 2) to pin the partition; verify with `nvidia-smi`'s utilization that no other process is using the GPUs. |
| `parse_warnings_any=True` | The log was truncated or `--log-to-console false` slipped through | Look at the offending log under `runs/`; rerun that single combo with `--force`. |
| `Status: A numerical error was encountered.` at N>=4 | Known sensitivity of distributed scaling for some MPS files | Lower `iteration_limit` to a value the problem reaches before going numerically unstable; or drop that problem from `problems.list`. |
| `validation error: distributed PDLP only supports DefaultPrecision` | `pdlp_precision` was set to single/mixed | Leave it at `pdlp_precision = 1` (double) in the params file. |
| `validation error: solve_lp_distributed_from_mps: presolve is not yet supported` | `presolve` is anything other than 0 (None) | Keep `presolve = 0` in the params file. |
| Sweep starts then `cudaErrorOperatingSystem` on rank 0 | No CUDA driver / running on a non-GPU host | Move to a GPU host. The driver fails fast; nothing else to do. |
| `setup_time_source = "first_iter_row"` everywhere | Binary is not the patched one | Rebuild `bullshit_mostovoi_cuopt` with the patch (already applied to source); the scaling result is still correct, just less precise. |

## What is NOT measured (and why)

- **End-to-end "real solve" wall-clock**: deliberately disabled by tightening
  tolerances. Iteration count is fixed instead. If you want to also measure
  time-to-optimality, run a second sweep with a different params file that
  drops the tight tolerances and removes `iteration_limit`.

## Quick reference

| File | Role | Produced by |
| --- | --- | --- |
| `problems.list` | input | you |
| `gpu_counts.list` | input | you |
| `config/distributed_pdlp.params` | input (cuopt_cli) | committed |
| `runs/*.log` | per-run logs | `run_sweep.sh` |
| `results.csv` | per-run data | `aggregate.py` |
| `results_summary.csv` | per-(problem, N) summary | `aggregate.py` |
| `plots/**.png` | charts | `plot_scaling.py` |
