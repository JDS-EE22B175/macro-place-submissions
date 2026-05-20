"""
Hybrid Analytical + Simulated Annealing Macro Placer (v2)

Approach:
  1. Extract net connectivity from PlacementCost for HPWL-aware optimization
  2. Smooth analytical phase: log-sum-exp wirelength with density spreading
     via Nesterov-accelerated gradient descent
  3. Legalize: greedy overlap resolution with minimum displacement
  4. SA refinement: congestion-aware simulated annealing using RUDY routing
     demand estimation alongside wirelength, with shift/swap/neighbor moves
  5. Evaluate final candidates via the actual proxy cost function and pick best

Usage:
    uv run evaluate submissions/my_placer/placer.py
    uv run evaluate submissions/my_placer/placer.py --all
    uv run evaluate submissions/my_placer/placer.py -b ibm01
"""

import math
import random
import sys
from pathlib import Path

import numpy as np
import torch

from macro_place.benchmark import Benchmark


# ---------------------------------------------------------------------------
# Helper: load PlacementCost for net extraction and cost evaluation
# ---------------------------------------------------------------------------

def _load_plc(name):
    """Load PlacementCost object for a given benchmark name."""
    from macro_place.loader import load_benchmark_from_dir, load_benchmark

    root = Path("external/MacroPlacement/Testcases/ICCAD04") / name
    if root.exists():
        _, plc = load_benchmark_from_dir(str(root))
        return plc

    ng45_map = {
        "ariane133_ng45": "ariane133",
        "ariane136_ng45": "ariane136",
        "nvdla_ng45": "nvdla",
        "mempool_tile_ng45": "mempool_tile",
    }
    d = ng45_map.get(name)
    if d:
        base = (
            Path("external/MacroPlacement/Flows/NanGate45")
            / d / "netlist" / "output_CT_Grouping"
        )
        if (base / "netlist.pb.txt").exists():
            _, plc = load_benchmark(
                str(base / "netlist.pb.txt"), str(base / "initial.plc")
            )
            return plc
    return None


# ---------------------------------------------------------------------------
# Helper: extract edge list (hard-macro-to-hard-macro connectivity)
# ---------------------------------------------------------------------------

def _extract_edges(benchmark, plc):
    """Build weighted edge list between hard macros from net connectivity."""
    name_to_bidx = {}
    for bidx, idx in enumerate(plc.hard_macro_indices):
        name_to_bidx[plc.modules_w_pins[idx].get_name()] = bidx

    edge_dict = {}
    for driver, sinks in plc.nets.items():
        macros = set()
        for pin in [driver] + sinks:
            parent = pin.split("/")[0]
            if parent in name_to_bidx:
                macros.add(name_to_bidx[parent])
        if len(macros) >= 2:
            ml = sorted(macros)
            w = 1.0 / (len(ml) - 1)
            for i in range(len(ml)):
                for j in range(i + 1, len(ml)):
                    pair = (ml[i], ml[j])
                    edge_dict[pair] = edge_dict.get(pair, 0) + w

    if not edge_dict:
        return np.zeros((0, 2), dtype=np.int64), np.zeros(0, dtype=np.float64)

    edges = np.array(list(edge_dict.keys()), dtype=np.int64)
    weights = np.array([edge_dict[tuple(e)] for e in edges], dtype=np.float64)
    return edges, weights


# ---------------------------------------------------------------------------
# RUDY congestion estimator (Rectangular Uniform wire DensitY)
# ---------------------------------------------------------------------------

def _rudy_congestion(pos, sizes, edges, edge_weights, cw, ch, grid_rows, grid_cols):
    """
    Estimate routing congestion using the RUDY model.

    For each edge, distributes wire demand uniformly across the bounding box
    of the two connected macros. Returns the average of the top-5% congested
    grid cells, which approximates the actual congestion cost.
    """
    if len(edges) == 0:
        return 0.0

    cell_w = cw / grid_cols
    cell_h = ch / grid_rows
    cong_grid = np.zeros((grid_rows, grid_cols), dtype=np.float64)

    for k in range(len(edges)):
        i, j = edges[k]
        # Bounding box of the two macro centers
        x_min = min(pos[i, 0], pos[j, 0])
        x_max = max(pos[i, 0], pos[j, 0])
        y_min = min(pos[i, 1], pos[j, 1])
        y_max = max(pos[i, 1], pos[j, 1])

        # Span in grid cells
        col_min = max(0, int(x_min / cell_w))
        col_max = min(grid_cols - 1, int(x_max / cell_w))
        row_min = max(0, int(y_min / cell_h))
        row_max = min(grid_rows - 1, int(y_max / cell_h))

        num_cells = max(1, (col_max - col_min + 1) * (row_max - row_min + 1))
        demand = edge_weights[k] / num_cells

        cong_grid[row_min:row_max + 1, col_min:col_max + 1] += demand

    # Also add macro blockage: macros block routing resources
    for i in range(len(pos)):
        hw, hh = sizes[i, 0] / 2, sizes[i, 1] / 2
        x_lo = max(0.0, pos[i, 0] - hw)
        x_hi = min(cw, pos[i, 0] + hw)
        y_lo = max(0.0, pos[i, 1] - hh)
        y_hi = min(ch, pos[i, 1] + hh)

        col_lo = max(0, int(x_lo / cell_w))
        col_hi = min(grid_cols - 1, int(x_hi / cell_w))
        row_lo = max(0, int(y_lo / cell_h))
        row_hi = min(grid_rows - 1, int(y_hi / cell_h))

        # Macro blocks routing tracks proportional to its area coverage
        for r in range(row_lo, row_hi + 1):
            for c in range(col_lo, col_hi + 1):
                # Fraction of cell covered by macro
                cx_lo = c * cell_w
                cx_hi = (c + 1) * cell_w
                cy_lo = r * cell_h
                cy_hi = (r + 1) * cell_h
                overlap_x = max(0, min(x_hi, cx_hi) - max(x_lo, cx_lo))
                overlap_y = max(0, min(y_hi, cy_hi) - max(y_lo, cy_lo))
                frac = (overlap_x * overlap_y) / (cell_w * cell_h)
                cong_grid[r, c] += frac * 0.5  # routing blockage factor

    # Top 5% congestion (mirrors the actual metric)
    flat = cong_grid.flatten()
    k = max(1, int(len(flat) * 0.05))
    top_k = np.partition(flat, -k)[-k:]
    return float(top_k.mean())


# ---------------------------------------------------------------------------
# Analytical phase: smooth wirelength + density penalty via gradient descent
# ---------------------------------------------------------------------------

def _analytical_place(pos, movable, sizes, half_w, half_h, cw, ch, n,
                      edges, edge_weights, num_iters=600):
    """
    Nesterov-accelerated gradient descent on smooth LSE wirelength + overlap
    penalty. Produces a good starting point for legalization.
    """
    if len(edges) == 0 or movable.sum() == 0:
        return pos

    pos_t = torch.tensor(pos, dtype=torch.float64, requires_grad=False)
    mov_idx = np.where(movable)[0]
    mov_pos = torch.tensor(pos[mov_idx], dtype=torch.float64, requires_grad=True)

    sizes_t = torch.tensor(sizes, dtype=torch.float64)
    edges_t = torch.tensor(edges, dtype=torch.long)
    ew_t = torch.tensor(edge_weights, dtype=torch.float64)
    hw_t = torch.tensor(half_w, dtype=torch.float64)
    hh_t = torch.tensor(half_h, dtype=torch.float64)

    gamma = max(cw, ch) * 0.02
    lr = max(cw, ch) * 0.003
    momentum = 0.9
    velocity = torch.zeros_like(mov_pos)

    density_weight_start = 0.01
    density_weight_end = 2.0

    best_pos = mov_pos.data.clone()
    best_cost = float("inf")

    for it in range(num_iters):
        frac = it / max(num_iters - 1, 1)
        lookahead = mov_pos.data + momentum * velocity
        mov_pos.data.copy_(lookahead)
        mov_pos.grad = None

        full_pos = pos_t.clone()
        full_pos[mov_idx] = mov_pos

        # Wirelength: log-sum-exp HPWL approximation
        src, dst = edges_t[:, 0], edges_t[:, 1]
        px_src, px_dst = full_pos[src, 0], full_pos[dst, 0]
        py_src, py_dst = full_pos[src, 1], full_pos[dst, 1]

        wl_x = gamma * (
            torch.logsumexp(torch.stack([px_src / gamma, px_dst / gamma]), dim=0)
            + torch.logsumexp(torch.stack([-px_src / gamma, -px_dst / gamma]), dim=0)
        )
        wl_y = gamma * (
            torch.logsumexp(torch.stack([py_src / gamma, py_dst / gamma]), dim=0)
            + torch.logsumexp(torch.stack([-py_src / gamma, -py_dst / gamma]), dim=0)
        )
        wl_cost = (ew_t * (wl_x + wl_y)).sum()

        # Pairwise overlap penalty (ramps up over iterations)
        density_weight = density_weight_start + frac * (density_weight_end - density_weight_start)
        mp = full_pos[:n]
        dx = mp[:, 0].unsqueeze(1) - mp[:, 0].unsqueeze(0)
        dy = mp[:, 1].unsqueeze(1) - mp[:, 1].unsqueeze(0)
        sep_x = (sizes_t[:, 0].unsqueeze(1) + sizes_t[:, 0].unsqueeze(0)) / 2 + 0.1
        sep_y = (sizes_t[:, 1].unsqueeze(1) + sizes_t[:, 1].unsqueeze(0)) / 2 + 0.1
        overlap_x = torch.clamp(sep_x - torch.abs(dx), min=0)
        overlap_y = torch.clamp(sep_y - torch.abs(dy), min=0)
        overlap_area = overlap_x * overlap_y
        diag_mask = 1.0 - torch.eye(n, dtype=torch.float64)
        density_cost = (overlap_area * diag_mask).sum() / 2.0

        total_cost = wl_cost + density_weight * density_cost
        total_cost.backward()

        with torch.no_grad():
            grad = mov_pos.grad
            if grad is not None:
                grad_norm = grad.norm()
                max_grad = max(cw, ch) * 0.5
                if grad_norm > max_grad:
                    grad = grad * (max_grad / grad_norm)
                velocity = momentum * velocity - lr * grad
                new_pos = lookahead + velocity
                for mi, gi in enumerate(mov_idx):
                    new_pos[mi, 0] = torch.clamp(new_pos[mi, 0], hw_t[gi], cw - hw_t[gi])
                    new_pos[mi, 1] = torch.clamp(new_pos[mi, 1], hh_t[gi], ch - hh_t[gi])
                mov_pos.data.copy_(new_pos)

            if wl_cost.item() < best_cost and density_cost.item() < 1e-3:
                best_cost = wl_cost.item()
                best_pos = mov_pos.data.clone()

        if frac > 0.7:
            lr *= 0.999

    result = pos.copy()
    final = best_pos if best_cost < float("inf") else mov_pos.data
    final_np = final.detach().numpy()
    for mi, gi in enumerate(mov_idx):
        result[gi] = final_np[mi]
    return result


# ---------------------------------------------------------------------------
# Legalization: resolve overlaps with minimum displacement
# ---------------------------------------------------------------------------

def _legalize(pos, movable, sizes, half_w, half_h, cw, ch, n):
    """Greedy legalization: place macros largest-first, spiral-search nearest legal spot."""
    sep_x = (sizes[:, 0:1] + sizes[:, 0:1].T) / 2
    sep_y = (sizes[:, 1:2] + sizes[:, 1:2].T) / 2
    order = sorted(range(n), key=lambda i: -sizes[i, 0] * sizes[i, 1])
    placed = np.zeros(n, dtype=bool)
    legal = pos.copy()

    for idx in order:
        if not movable[idx]:
            placed[idx] = True
            continue
        if placed.any():
            dx = np.abs(legal[idx, 0] - legal[:, 0])
            dy = np.abs(legal[idx, 1] - legal[:, 1])
            c = (dx < sep_x[idx] + 0.05) & (dy < sep_y[idx] + 0.05) & placed
            c[idx] = False
            if not c.any():
                placed[idx] = True
                continue

        step = max(sizes[idx, 0], sizes[idx, 1]) * 0.2
        best_p = legal[idx].copy()
        best_d = float("inf")
        for r in range(1, 200):
            found = False
            for dxm in range(-r, r + 1):
                for dym in range(-r, r + 1):
                    if abs(dxm) != r and abs(dym) != r:
                        continue
                    cx = np.clip(pos[idx, 0] + dxm * step, half_w[idx], cw - half_w[idx])
                    cy = np.clip(pos[idx, 1] + dym * step, half_h[idx], ch - half_h[idx])
                    if placed.any():
                        ddx = np.abs(cx - legal[:, 0])
                        ddy = np.abs(cy - legal[:, 1])
                        c = (ddx < sep_x[idx] + 0.05) & (ddy < sep_y[idx] + 0.05) & placed
                        c[idx] = False
                        if c.any():
                            continue
                    d = (cx - pos[idx, 0]) ** 2 + (cy - pos[idx, 1]) ** 2
                    if d < best_d:
                        best_d = d
                        best_p = np.array([cx, cy])
                        found = True
                if found and best_d < (step * r * 0.5) ** 2:
                    break
            if found:
                break
        legal[idx] = best_p
        placed[idx] = True
    return legal


# ---------------------------------------------------------------------------
# SA refinement: congestion-aware simulated annealing
# ---------------------------------------------------------------------------

def _sa_refine(pos, edges, edge_weights, movable, sizes, half_w, half_h,
               cw, ch, n, grid_rows, grid_cols, num_iters=5000):
    """
    SA with shift, swap, and neighbor-attract moves.
    Cost function blends wirelength + RUDY congestion estimate.
    """
    movable_idx = np.where(movable)[0]
    if len(movable_idx) == 0 or len(edges) == 0:
        return pos

    pos = pos.copy()
    sep_x = (sizes[:, 0:1] + sizes[:, 0:1].T) / 2
    sep_y = (sizes[:, 1:2] + sizes[:, 1:2].T) / 2

    # Neighbor lists for connectivity-aware moves
    neighbors = [[] for _ in range(n)]
    for i, j in edges:
        neighbors[i].append(j)
        neighbors[j].append(i)

    def wl_cost():
        dx = np.abs(pos[edges[:, 0], 0] - pos[edges[:, 1], 0])
        dy = np.abs(pos[edges[:, 0], 1] - pos[edges[:, 1], 1])
        return (edge_weights * (dx + dy)).sum()

    def combined_cost():
        """Wirelength + RUDY congestion, weighted to match proxy cost formula."""
        wl = wl_cost()
        rudy = _rudy_congestion(pos, sizes, edges, edge_weights,
                                cw, ch, grid_rows, grid_cols)
        return wl + 0.5 * rudy

    def check_overlap(idx):
        gap = 0.05
        ddx = np.abs(pos[idx, 0] - pos[:, 0])
        ddy = np.abs(pos[idx, 1] - pos[:, 1])
        overlaps = (ddx < sep_x[idx] + gap) & (ddy < sep_y[idx] + gap)
        overlaps[idx] = False
        return overlaps.any()

    # Evaluate initial cost with congestion every N steps to keep it tractable.
    # For per-move decisions, use the cheaper wirelength-only cost but
    # periodically recompute the full cost to track best.
    current_wl = wl_cost()
    best_pos = pos.copy()
    best_full_cost = combined_cost()

    T_start = max(cw, ch) * 0.12
    T_end = max(cw, ch) * 0.0005

    # Recompute full cost every this many steps
    full_eval_interval = max(50, num_iters // 100)

    for step in range(num_iters):
        frac = step / num_iters
        T = T_start * (T_end / T_start) ** frac

        move_type = random.random()
        i = random.choice(movable_idx)
        old_x, old_y = pos[i, 0], pos[i, 1]

        if move_type < 0.45:
            # SHIFT
            shift_scale = T * (0.3 + 0.7 * (1 - frac))
            pos[i, 0] = np.clip(pos[i, 0] + random.gauss(0, shift_scale),
                                half_w[i], cw - half_w[i])
            pos[i, 1] = np.clip(pos[i, 1] + random.gauss(0, shift_scale),
                                half_h[i], ch - half_h[i])

        elif move_type < 0.75:
            # SWAP
            if neighbors[i] and random.random() < 0.6:
                cands = [j for j in neighbors[i] if movable[j]]
                j = random.choice(cands) if cands else random.choice(movable_idx)
            else:
                j = random.choice(movable_idx)

            if i != j:
                old_jx, old_jy = pos[j, 0], pos[j, 1]
                pos[i, 0] = np.clip(old_jx, half_w[i], cw - half_w[i])
                pos[i, 1] = np.clip(old_jy, half_h[i], ch - half_h[i])
                pos[j, 0] = np.clip(old_x, half_w[j], cw - half_w[j])
                pos[j, 1] = np.clip(old_y, half_h[j], ch - half_h[j])

                if check_overlap(i) or check_overlap(j):
                    pos[i, 0] = old_x; pos[i, 1] = old_y
                    pos[j, 0] = old_jx; pos[j, 1] = old_jy
                    continue

                new_wl = wl_cost()
                delta = new_wl - current_wl
                if delta < 0 or random.random() < math.exp(-delta / max(T, 1e-10)):
                    current_wl = new_wl
                else:
                    pos[i, 0] = old_x; pos[i, 1] = old_y
                    pos[j, 0] = old_jx; pos[j, 1] = old_jy

                # Track best using full cost at intervals
                if step % full_eval_interval == 0:
                    fc = combined_cost()
                    if fc < best_full_cost:
                        best_full_cost = fc
                        best_pos = pos.copy()
                continue

        else:
            # MOVE TOWARD NEIGHBOR
            if neighbors[i]:
                j = random.choice(neighbors[i])
                alpha = random.uniform(0.05, 0.35)
                pos[i, 0] = np.clip(pos[i, 0] + alpha * (pos[j, 0] - pos[i, 0]),
                                    half_w[i], cw - half_w[i])
                pos[i, 1] = np.clip(pos[i, 1] + alpha * (pos[j, 1] - pos[i, 1]),
                                    half_h[i], ch - half_h[i])

        # Check overlap for shift/neighbor moves
        if check_overlap(i):
            pos[i, 0] = old_x
            pos[i, 1] = old_y
            continue

        new_wl = wl_cost()
        delta = new_wl - current_wl
        if delta < 0 or random.random() < math.exp(-delta / max(T, 1e-10)):
            current_wl = new_wl
        else:
            pos[i, 0] = old_x
            pos[i, 1] = old_y

        # Periodic full cost evaluation for best tracking
        if step % full_eval_interval == 0:
            fc = combined_cost()
            if fc < best_full_cost:
                best_full_cost = fc
                best_pos = pos.copy()

    # Final check: is the last position better than tracked best?
    fc = combined_cost()
    if fc < best_full_cost:
        best_pos = pos.copy()

    return best_pos


# ---------------------------------------------------------------------------
# Evaluate a candidate using actual proxy cost (for picking best result)
# ---------------------------------------------------------------------------

def _eval_proxy(pos_hard, benchmark, plc):
    """Compute actual proxy cost for a hard macro placement."""
    from macro_place.objective import compute_proxy_cost

    full_pos = benchmark.macro_positions.clone()
    full_pos[:benchmark.num_hard_macros] = torch.tensor(pos_hard, dtype=torch.float32)
    costs = compute_proxy_cost(full_pos, benchmark, plc)
    return costs["proxy_cost"], costs["overlap_count"]


# ---------------------------------------------------------------------------
# Main placer class
# ---------------------------------------------------------------------------

class HybridAnalyticalSAPlacer:
    """
    Hybrid macro placer: analytical optimization -> legalization -> SA polish.

    Two candidate paths (analytical start vs initial start) are evaluated using
    the actual TILOS proxy cost, and the better result is returned.
    """

    def __init__(self, seed=42, analytical_iters=600, sa_iters=6000):
        self.seed = seed
        self.analytical_iters = analytical_iters
        self.sa_iters = sa_iters

    def place(self, benchmark: Benchmark) -> torch.Tensor:
        torch.manual_seed(self.seed)
        random.seed(self.seed)
        np.random.seed(self.seed)

        n_hard = benchmark.num_hard_macros
        sizes_np = benchmark.macro_sizes[:n_hard].numpy().astype(np.float64)
        cw = float(benchmark.canvas_width)
        ch = float(benchmark.canvas_height)
        half_w = sizes_np[:, 0] / 2
        half_h = sizes_np[:, 1] / 2
        movable = benchmark.get_movable_mask()[:n_hard].numpy()

        plc = _load_plc(benchmark.name)
        if plc is not None:
            edges, edge_weights = _extract_edges(benchmark, plc)
        else:
            edges = np.zeros((0, 2), dtype=np.int64)
            edge_weights = np.zeros(0, dtype=np.float64)

        pos = benchmark.macro_positions[:n_hard].numpy().copy().astype(np.float64)
        grid_rows = benchmark.grid_rows
        grid_cols = benchmark.grid_cols

        candidates = []

        # --- Path A: analytical -> legalize -> SA ---
        if len(edges) > 0:
            random.seed(self.seed)
            np.random.seed(self.seed)
            pos_a = _analytical_place(
                pos, movable, sizes_np, half_w, half_h, cw, ch, n_hard,
                edges, edge_weights, num_iters=self.analytical_iters,
            )
            pos_a = _legalize(pos_a, movable, sizes_np, half_w, half_h, cw, ch, n_hard)
            random.seed(self.seed + 1)
            pos_a = _sa_refine(
                pos_a, edges, edge_weights, movable, sizes_np,
                half_w, half_h, cw, ch, n_hard, grid_rows, grid_cols,
                num_iters=self.sa_iters,
            )
            candidates.append(("analytical+SA", pos_a))

        # --- Path B: initial -> legalize -> SA ---
        pos_b = _legalize(pos, movable, sizes_np, half_w, half_h, cw, ch, n_hard)
        if len(edges) > 0:
            random.seed(self.seed + 2)
            pos_b = _sa_refine(
                pos_b, edges, edge_weights, movable, sizes_np,
                half_w, half_h, cw, ch, n_hard, grid_rows, grid_cols,
                num_iters=self.sa_iters,
            )
        candidates.append(("initial+SA", pos_b))

        # Pick best candidate by actual proxy cost
        best_name = None
        best_cost = float("inf")
        best_hard = candidates[0][1]  # fallback

        if plc is not None:
            for name, cand in candidates:
                cost, overlaps = _eval_proxy(cand, benchmark, plc)
                if overlaps == 0 and cost < best_cost:
                    best_cost = cost
                    best_hard = cand
                    best_name = name
        else:
            # No plc available, pick by edge wirelength
            def edge_wl(p):
                if len(edges) == 0:
                    return 0.0
                dx = np.abs(p[edges[:, 0], 0] - p[edges[:, 1], 0])
                dy = np.abs(p[edges[:, 0], 1] - p[edges[:, 1], 1])
                return (edge_weights * (dx + dy)).sum()
            best_hard = min(candidates, key=lambda c: edge_wl(c[1]))[1]

        # Build final placement
        full_pos = benchmark.macro_positions.clone()
        full_pos[:n_hard] = torch.tensor(best_hard, dtype=torch.float32)

        # -----------------------------------------------------------------------
        # Soft macro optimization (Priority 2)
        # -----------------------------------------------------------------------
        if plc is not None:
            from macro_place.objective import _set_placement
            # Sync our best hard macro placement into plc
            _set_placement(plc, full_pos, benchmark)
            
            # Run force-directed standard cell optimization
            canvas_size = max(cw, ch)
            plc.optimize_stdcells(
                use_current_loc=False, move_stdcells=True, move_macros=False,
                log_scale_conns=False, use_sizes=False, io_factor=1.0,
                num_steps=[100, 100, 100],
                max_move_distance=[canvas_size/100]*3,
                attract_factor=[100, 1.0e-3, 1.0e-5],
                repel_factor=[0, 1.0e6, 1.0e7],
            )
            
            # Extract the newly optimized soft macro positions back into full_pos
            for i, macro_idx in enumerate(benchmark.soft_macro_indices):
                node = plc.modules_w_pins[macro_idx]
                x, y = node.get_pos()
                full_pos[n_hard + i, 0] = float(x)
                full_pos[n_hard + i, 1] = float(y)

        return full_pos
