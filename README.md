# Hybrid Analytical + Simulated Annealing Macro Placer (HybridASA)

This repository contains our submission for the PARTCL Macro Placement Challenge.

## Approach & Methodology
Our approach combines global analytical placement with a highly optimized simulated annealing (SA) refinement phase. It is designed to strictly adhere to the zero-overlap constraint while heavily optimizing for proxy cost within the allowed 1-hour time budget.

The algorithm runs in multiple phases:
1. **Connectivity Extraction**: We parse the `.plc` definitions to extract a hard-macro-to-hard-macro adjacency edge list, weighted by HPWL connectivity.
2. **Analytical Global Placement**: We apply a smooth, differentiable approximation of HPWL (using log-sum-exp) combined with a pairwise density penalty. This is optimized using Nesterov-accelerated gradient descent to quickly generate a highly competitive starting candidate.
3. **Legalization**: A fast, greedy legalizer places macros largest-first, spiraling outward to find the nearest legal spot.
4. **Time-Budgeted Simulated Annealing Refinement**: The core of the optimization. We utilize an **incremental O(degree)** wirelength cost evaluation—meaning each SA move only recalculates the cost of edges directly touching the moved macro, rather than the entire `O(E)` graph. This enables millions of SA iterations. The SA phase uses shift, swap, and connectivity-aware neighbor moves. The number of iterations is completely dynamic and scales automatically to fill the remaining time out of the 1-hour budget, safely returning the best found position when the hard stop is reached.
5. **Soft Macro Optimization**: If sufficient time remains in the budget, we run a short force-directed placement step to optimize standard cell/soft-macro locations around the locked hard macros.

## Execution
The standard single-benchmark evaluation works normally. The total execution time budget is hard-coded to scale up to ~59.0 minutes to ensure maximum optimization without timing out on the judges' evaluation hardware.

```bash
uv run evaluate my_placer/placer.py -b ibm01
```

### Parallel Evaluation Wrapper
We have also included a custom wrapper (`parallel_evaluate.py`) that effortlessly evaluates the benchmark suites concurrently. It utilizes up to 8 parallel background processes to ensure that evaluating the full suite completes in under 3 hours, while still granting each benchmark its full 60-minute optimization window.

To evaluate all 17 IBM benchmarks in parallel:
```bash
python my_placer/parallel_evaluate.py my_placer/placer.py --all
```

To evaluate the NG45 commercial designs in parallel:
```bash
python my_placer/parallel_evaluate.py my_placer/placer.py --ng45
```

The script dynamically saves full evaluation logs to the `eval_results/` directory and prints the final `proxy=` scores cleanly to standard output as each benchmark completes.
