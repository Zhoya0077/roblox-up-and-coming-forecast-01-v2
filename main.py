"""Roblox Up-and-Coming chart forecaster v2: empirical rank-transition matrices.

Reads /input/history.ndjson + task.json, writes forecast.json with per-place
probabilities and a top10. Model: the chart is a Markov chain over rank
states (exact ranks 1..places, coarser bands beyond, plus ABSENT). One
transition matrix T(dt) is estimated per lag bucket from ALL read pairs in
the history, shrunk toward a diffusion prior fitted to the same pairs.
For each game and each scored slot, the row of T(h) for the game's
last-known state (optionally drift-shifted by recent rank velocity) gives
the place distribution; averaging over slots gives the forecast. Mass for
never-before-seen games is reserved as unnamed share via the empirical
never-seen-share curve (same idea as v1). Fully deterministic, no RNG.

Stdlib only, no network. Reads window/places from task.json, never
hardcodes dates. Handles null details (archive reads) gracefully.
"""
import json
import math
import os

# --- tunable constants (tuned on held-out cuts only, never the practice window) ---
ALPHA = 80.0        # pseudocount weight of the diffusion prior per matrix row
DRIFT_SHRINK = 0.9  # rank-velocity drift shrink; velocity in ranks/hour
DRIFT_TAU_H = 24.0  # drift decay horizon
DRIFT_CAP = 0.5     # ranks/hour cap on shrunk velocity
RESERVE_SCALE = 1.4  # scale on the empirical newcomer-share reserve
MIN_UNNAMED = 0.02
MAX_PAIRS = 6000    # subsample read pairs evenly above this (dense histories)
# Lag bucket edges (hours) for T(dt) estimation.
BUCKET_EDGES = [0, 4, 7, 10, 14, 20, 28, 40, 56, 76, 110, 160, 240, 400]
# Rank bands beyond the exact top `places` states.
BANDS = [(11, 15), (16, 20), (21, 30), (31, 45), (46, 60), (61, 80), (81, 10 ** 9)]
BANDS_FINE = [(11, 12), (13, 15), (16, 20), (21, 25), (26, 30), (31, 40),
              (41, 55), (56, 70), (71, 85), (86, 10 ** 9)]
USE_FINE_BANDS = True
ABSENT_SPLIT = 1    # 1 = single ABSENT state; 3 = split by last-seen rank band
INTERP = "linear"   # "linear" or "log" interpolation between bucket matrices
NAMING_MULT = 2.0   # water-fill threshold multiplier (name while p > mult*per-game)
POWER = 1.15        # calibration power on place probs (1 = none), renormalized per place
RECENCY_TAU = None  # hours; if set, pair weights decay exp(-(t_last-t_j)/tau)
RESERVE_PER_PLACE = False  # estimate the newcomer reserve separately per place
RESERVE_PLACE_SHRINK = 0.5  # shrink of per-place reserve toward the uniform reserve


def parse_time(s):
    return _iso_to_epoch(s) / 3600.0


def _iso_to_epoch(s):
    from datetime import datetime, timezone

    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    return datetime.fromisoformat(s).timestamp()


def load_reads(input_dir):
    with open(os.path.join(input_dir, "task.json")) as f:
        task = json.load(f)
    reads = []
    with open(os.path.join(input_dir, "history.ndjson")) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            chart = {}
            for g in r.get("up_and_coming") or []:
                chart[g["universe_id"]] = g["rank"]
            created = {}
            details = r.get("details")
            if details:
                for uid_s, d in details.items():
                    try:
                        uid = int(uid_s)
                    except (TypeError, ValueError):
                        continue
                    if d and d.get("created"):
                        created[uid] = parse_time(d["created"])
            reads.append({"t": parse_time(r["at"]), "chart": chart, "created": created})
    reads.sort(key=lambda r: r["t"])
    return task, reads


def bands():
    return BANDS_FINE if USE_FINE_BANDS else BANDS


def n_absent_states():
    return 3 if ABSENT_SPLIT == 3 else 1


def n_states_for(places):
    return places + len(bands()) + n_absent_states()


def absent_base(places):
    return places + len(bands())


def absent_state_for(last_rank, places):
    """ABSENT state index by last-seen rank band (single state if no split)."""
    if ABSENT_SPLIT != 3:
        return absent_base(places)
    if last_rank <= 10:
        return absent_base(places)
    if last_rank <= 30:
        return absent_base(places) + 1
    return absent_base(places) + 2


def is_absent_state(state, places):
    return state >= absent_base(places)


def state_of(rank, places):
    """Rank -> state index: 0..places-1 exact, then bands."""
    if rank <= places:
        return rank - 1
    for idx, (lo, hi) in enumerate(bands()):
        if lo <= rank <= hi:
            return places + idx
    return places + len(bands()) - 1


def state_bounds(state, places, chart_size):
    """(lo_rank, hi_rank, midpoint) covered by a state; ABSENT -> None."""
    if state < places:
        return (state + 1, state + 1, float(state + 1))
    idx = state - places
    if idx < len(bands()):
        lo, hi = bands()[idx]
        hi = min(hi, chart_size)
        return (lo, hi, (lo + hi) / 2.0)
    return None


def newcomer_share_curve(reads):
    """Share of top-10 slots held by never-before-seen games, vs horizon."""
    n = len(reads)
    min_i = min(10, max(0, n // 3))
    buckets = {}
    for i in range(min_i, n):
        seen = set()
        for k in range(i):
            seen |= set(reads[k]["chart"].keys())
        for j in range(i + 1, n):
            dt = reads[j]["t"] - reads[i]["t"]
            if dt <= 0:
                continue
            b = max(6, int(round(dt / 12.0)) * 12)
            acc = buckets.setdefault(b, [0, 0])
            top = sorted(reads[j]["chart"], key=lambda u: reads[j]["chart"][u])[:10]
            for uid in top:
                acc[1] += 1
                if uid not in seen:
                    acc[0] += 1
    pts = sorted((b, a[0] / a[1]) for b, a in buckets.items() if a[1] >= 40)
    return pts


def _curve_at(pts, h):
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    if h <= xs[0]:
        return ys[0] * (h / xs[0])
    if h >= xs[-1]:
        return ys[-1]
    for a in range(len(xs) - 1):
        if xs[a] <= h <= xs[a + 1]:
            f = (h - xs[a]) / (xs[a + 1] - xs[a])
            return ys[a] + f * (ys[a + 1] - ys[a])
    return ys[-1]


def newcomer_reserve(reads, horizons):
    """Mean never-seen share over the given horizons (hours after last read)."""
    pts = newcomer_share_curve(reads)
    if len(pts) < 3:
        return 0.08
    mean_share = sum(_curve_at(pts, h) for h in horizons) / len(horizons) if horizons else 0.08
    return min(max(mean_share * RESERVE_SCALE, MIN_UNNAMED), 0.30)


def newcomer_share_curve_per_place(reads, places):
    """Per-place never-seen share curves: {place: [(dt_bucket, share)]}."""
    n = len(reads)
    min_i = min(10, max(0, n // 3))
    buckets = {}
    for i in range(min_i, n):
        seen = set()
        for k in range(i):
            seen |= set(reads[k]["chart"].keys())
        for j in range(i + 1, n):
            dt = reads[j]["t"] - reads[i]["t"]
            if dt <= 0:
                continue
            b = max(6, int(round(dt / 12.0)) * 12)
            chart_j = sorted(reads[j]["chart"], key=lambda u: reads[j]["chart"][u])
            for k in range(1, places + 1):
                if k > len(chart_j):
                    break
                acc = buckets.setdefault((b, k), [0, 0])
                acc[1] += 1
                if chart_j[k - 1] not in seen:
                    acc[0] += 1
    curves = {}
    for k in range(1, places + 1):
        pts = sorted((b, a[0] / a[1]) for (b, kk), a in buckets.items() if kk == k and a[1] >= 20)
        if len(pts) >= 3:
            curves[k] = pts
    return curves


def newcomer_reserves(reads, horizons, places):
    """Reserve per place (list index 0..places-1). Uniform unless enabled."""
    uni = newcomer_reserve(reads, horizons)
    if not RESERVE_PER_PLACE:
        return [uni] * places
    curves = newcomer_share_curve_per_place(reads, places)
    out = []
    for k in range(1, places + 1):
        pts = curves.get(k)
        if pts is None:
            out.append(uni)
            continue
        mean_share = sum(_curve_at(pts, h) for h in horizons) / len(horizons) if horizons else 0.08
        per = min(max(mean_share * RESERVE_SCALE, MIN_UNNAMED), 0.30)
        out.append(RESERVE_PLACE_SHRINK * uni + (1.0 - RESERVE_PLACE_SHRINK) * per)
    return out


def _norm_cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def fit_drift_curves(reads):
    """Empirical mean/std of rank change and survival vs lag, for the prior.

    Returns (lags, mean_change, std_change, survival) sorted by lag, from
    pooled read pairs in coarse buckets.
    """
    buckets = {}
    n = len(reads)
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    if len(pairs) > MAX_PAIRS:
        step = len(pairs) / MAX_PAIRS
        pairs = [pairs[int(k * step)] for k in range(MAX_PAIRS)]
    for i, j in pairs:
        dt = reads[j]["t"] - reads[i]["t"]
        if dt <= 0:
            continue
        b = max(6, int(round(dt / 6.0)) * 6)
        acc = buckets.setdefault(b, [0, 0, 0.0, 0.0])
        ci, cj = reads[i]["chart"], reads[j]["chart"]
        for uid, rk in ci.items():
            acc[0] += 1
            if uid in cj:
                acc[1] += 1
                d = cj[uid] - rk
                acc[2] += d
                acc[3] += d * d
    lags, means, stds, survs = [], [], [], []
    for b in sorted(buckets):
        cnt, s, s1, s2 = buckets[b]
        if cnt < 30 or s < 30:
            continue
        mean = s1 / s
        var = max(s2 / s - mean * mean, 0.25)
        lags.append(float(b))
        means.append(mean)
        stds.append(math.sqrt(var))
        survs.append(s / cnt)
    return lags, means, stds, survs


def _interp(x, xs, ys, left=None, right=None):
    if not xs:
        return left if left is not None else 0.0
    if x <= xs[0]:
        return ys[0] if left is None else left + (ys[0] - left) * (x / xs[0] if xs[0] else 1.0)
    if x >= xs[-1]:
        return ys[-1] if right is None else right
    for a in range(len(xs) - 1):
        if xs[a] <= x <= xs[a + 1]:
            f = (x - xs[a]) / (xs[a + 1] - xs[a])
            return ys[a] + f * (ys[a + 1] - ys[a])
    return ys[-1]


def diffusion_prior_row(from_state, dt, places, n_states,
                        chart_size, lags, means, stds, survs, marginal):
    """Prior transition row: Gaussian rank diffusion + empirical exit mass."""
    if is_absent_state(from_state, places):
        # Absent games re-enter like the marginal next-state distribution,
        # but keep most mass absent (re-entry is rare per unit time).
        # Re-enter mass grows with dt toward the marginal.
        reenter = min(max(0.35 * (1.0 - math.exp(-dt / 48.0)), 0.02), 0.6)
        tot = sum(marginal)
        row = [r / tot * reenter if tot > 0 else 0.0 for r in marginal]
        for s in range(absent_base(places), n_states):
            row[s] = 0.0
        row[from_state] = 1.0 - reenter
        return row
    b = state_bounds(from_state, places, chart_size)
    mid = b[2]
    mean_d = _interp(dt, lags, means, left=0.0)
    sd = _interp(dt, lags, stds, left=1.0)
    sd = max(sd, 1.0)
    surv = _interp(dt, lags, survs, left=1.0)
    surv = min(max(surv, 0.05), 1.0)
    mu = mid + mean_d
    row = [0.0] * n_states
    total = 0.0
    for s in range(absent_base(places)):
        lo, hi, _ = state_bounds(s, places, chart_size)
        p = _norm_cdf((hi + 0.5 - mu) / sd) - _norm_cdf((lo - 0.5 - mu) / sd)
        row[s] = max(p, 0.0)
        total += row[s]
    if total > 0:
        for s in range(absent_base(places)):
            row[s] = row[s] / total * surv
    # Exiting mass goes to the absent state matching the from-state's band.
    row[absent_state_for(int(round(mid)), places)] = 1.0 - surv
    return row


def estimate_matrices(reads, places):
    """Estimate T(dt) per lag bucket. Returns (centers, matrices, diag)."""
    n_states = n_states_for(places)
    sizes = [len(r["chart"]) for r in reads]
    chart_size = sorted(sizes)[len(sizes) // 2] if sizes else 97
    lags, means, stds, survs = fit_drift_curves(reads)
    # Marginal next-state distribution (for the ABSENT prior rows).
    marg_counts = [0.0] * n_states
    for r in reads:
        for rk in r["chart"].values():
            marg_counts[state_of(rk, places)] += 1
    tot = sum(marg_counts)
    marginal = [c / tot if tot else 1.0 / n_states for c in marg_counts]
    # First-seen index and running last-seen rank per game.
    first_seen = {}
    for idx, r in enumerate(reads):
        for uid in r["chart"]:
            if uid not in first_seen:
                first_seen[uid] = idx
    last_seen_rank = {}
    last_seen_upto = []  # snapshot of last_seen_rank before each read index
    for idx, r in enumerate(reads):
        last_seen_upto.append(dict(last_seen_rank))
        for uid, rk in r["chart"].items():
            last_seen_rank[uid] = rk
    t_last = reads[-1]["t"]
    n = len(reads)
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    if len(pairs) > MAX_PAIRS:
        step = len(pairs) / MAX_PAIRS
        pairs = [pairs[int(k * step)] for k in range(MAX_PAIRS)]
    n_buckets = len(BUCKET_EDGES) - 1
    counts = [[[0.0] * n_states for _ in range(n_states)] for _ in range(n_buckets)]
    dt_sums = [0.0] * n_buckets
    dt_ns = [0] * n_buckets
    for i, j in pairs:
        dt = reads[j]["t"] - reads[i]["t"]
        if dt <= 0:
            continue
        b = None
        for k in range(n_buckets):
            if BUCKET_EDGES[k] < dt <= BUCKET_EDGES[k + 1]:
                b = k
                break
        if b is None:
            continue
        w = 1.0
        if RECENCY_TAU:
            w = math.exp(-(t_last - reads[j]["t"]) / RECENCY_TAU)
        dt_sums[b] += dt
        dt_ns[b] += 1
        ci, cj = reads[i]["chart"], reads[j]["chart"]
        for uid, rk in ci.items():
            s = state_of(rk, places)
            if uid in cj:
                s2 = state_of(cj[uid], places)
            else:
                s2 = absent_state_for(rk, places)
            counts[b][s][s2] += w
        # ABSENT at i (seen before i) -> state at j.
        seen_before = last_seen_upto[i]
        for uid, lrk in seen_before.items():
            if uid not in ci:
                s = absent_state_for(lrk, places)
                if uid in cj:
                    s2 = state_of(cj[uid], places)
                else:
                    s2 = absent_state_for(lrk, places)
                counts[b][s][s2] += w
    centers = []
    matrices = []
    for b in range(n_buckets):
        if dt_ns[b] == 0:
            continue
        center = dt_sums[b] / dt_ns[b]
        mat = []
        for s in range(n_states):
            prior = diffusion_prior_row(s, center, places, n_states,
                                        chart_size, lags, means, stds, survs, marginal)
            row_counts = counts[b][s]
            n_obs = sum(row_counts)
            row = [(row_counts[t] + ALPHA * prior[t]) / (n_obs + ALPHA)
                   for t in range(n_states)]
            mat.append(row)
        centers.append(center)
        matrices.append(mat)
    diag = {"n_states": n_states, "chart_size": chart_size,
            "bucket_centers": [round(c, 1) for c in centers],
            "bucket_pairs": dt_ns, "n_pairs": len(pairs)}
    return centers, matrices, diag


def matrix_at(h, centers, matrices, n_states):
    """T(h) by interpolation between bucket matrices (identity below first)."""
    if not centers:
        return [[1.0 if a == b else 0.0 for b in range(n_states)] for a in range(n_states)]
    if h <= centers[0]:
        w = h / centers[0] if centers[0] > 0 else 1.0
        base = matrices[0]
        return [[(1.0 - w) * (1.0 if a == b else 0.0) + w * base[a][b]
                 for b in range(n_states)] for a in range(n_states)]
    if h >= centers[-1]:
        return matrices[-1]
    for k in range(len(centers) - 1):
        if centers[k] <= h <= centers[k + 1]:
            if INTERP == "log" and centers[k] > 0:
                w = (math.log(h) - math.log(centers[k])) / (math.log(centers[k + 1]) - math.log(centers[k]))
            else:
                w = (h - centers[k]) / (centers[k + 1] - centers[k])
            a_mat, b_mat = matrices[k], matrices[k + 1]
            return [[(1.0 - w) * a_mat[a][b] + w * b_mat[a][b]
                     for b in range(n_states)] for a in range(n_states)]
    return matrices[-1]


def rank_velocities(reads, last_idx):
    """Shrunk recent rank velocity (ranks/hour, negative = improving) per game."""
    vel = {}
    if len(reads) < 3:
        return vel
    k0 = max(0, len(reads) - 5)
    for uid in reads[last_idx]["chart"]:
        pts = [(reads[i]["t"], reads[i]["chart"][uid]) for i in range(k0, len(reads)) if uid in reads[i]["chart"]]
        if len(pts) < 3:
            continue
        n = len(pts)
        mt = sum(p[0] for p in pts) / n
        mr = sum(p[1] for p in pts) / n
        den = sum((p[0] - mt) ** 2 for p in pts)
        if den <= 0:
            continue
        slope = sum((p[0] - mt) * (p[1] - mr) for p in pts) / den
        v = DRIFT_SHRINK * slope
        vel[uid] = max(-DRIFT_CAP, min(DRIFT_CAP, v))
    return vel


def build_forecast(task, reads, n_sims=None, seed=None):
    places = int(task["places"])
    win_from = parse_time(task["window"]["from"])
    win_until = parse_time(task["window"]["until"])
    slots = int(task["window"]["slots"])
    slot_starts = [win_from + h * (win_until - win_from) / slots for h in range(slots)]
    t0 = reads[-1]["t"]
    horizons = [s - t0 for s in slot_starts]

    centers, matrices, diag = estimate_matrices(reads, places)
    n_states = n_states_for(places)
    chart_size = diag["chart_size"]

    last_chart = reads[-1]["chart"]
    all_uids = sorted({uid for r in reads for uid in r["chart"]})
    vel = rank_velocities(reads, len(reads) - 1)
    last_seen_rank = {}
    for r in reads:
        for uid, rk in r["chart"].items():
            last_seen_rank[uid] = rk

    # Starting state per game (last read), with optional drift-shifted rank.
    start_state = {}
    for uid in all_uids:
        if uid in last_chart:
            start_state[uid] = state_of(last_chart[uid], places)
        else:
            start_state[uid] = absent_state_for(last_seen_rank[uid], places)

    # Average place distribution over slots.
    probs = {}  # (uid, place) -> p
    for h in horizons:
        mat = matrix_at(h, centers, matrices, n_states)
        for uid in all_uids:
            s = start_state[uid]
            if uid in vel:
                # Drift-shift the starting rank, then take that state's row.
                r0 = last_chart[uid]
                shift = vel[uid] * DRIFT_TAU_H * (1.0 - math.exp(-max(h, 0.0) / DRIFT_TAU_H))
                r_eff = r0 + shift
                r_eff = max(1.0, min(float(chart_size + 30), r_eff))
                s = state_of(int(round(r_eff)), places)
            row = mat[s]
            for k in range(places):
                p = row[k]
                if p > 0:
                    key = (uid, k + 1)
                    probs[key] = probs.get(key, 0.0) + p
    n_slots = float(len(horizons)) if horizons else 1.0
    for key in probs:
        probs[key] /= n_slots

    # Optional calibration power, renormalized per place to the same total.
    if POWER != 1.0:
        totals = {}
        for (uid, pl), p in probs.items():
            totals[pl] = totals.get(pl, 0.0) + p
        powered = {}
        ptot = {}
        for (uid, pl), p in probs.items():
            q = p ** POWER
            powered[(uid, pl)] = q
            ptot[pl] = ptot.get(pl, 0.0) + q
        for key, q in powered.items():
            pl = key[1]
            probs[key] = q * totals[pl] / ptot[pl] if ptot[pl] > 0 else q

    reserves = newcomer_reserves(reads, horizons, places)
    diag["reserve"] = reserves
    diag["alpha"] = ALPHA
    diag["drift_shrink"] = DRIFT_SHRINK

    probabilities = []
    for k in range(1, places + 1):
        reserve = reserves[k - 1]
        cands = sorted(
            ((p, uid) for (uid, pl), p in probs.items() if pl == k),
            key=lambda x: (-x[0], x[1]),
        )
        named = []
        total = 0.0
        for p, uid in cands:
            unnamed = 1.0 - total
            per_game = unnamed / max(chart_size - len(named), 1)
            if p <= NAMING_MULT * per_game:
                break
            if total + p > 1.0 - reserve:
                break
            named.append((uid, p))
            total += p
        if len(named) > 100:
            named = named[:100]
            total = sum(p for _, p in named)
        for uid, p in named:
            probabilities.append({"universe_id": uid, "place": k, "p": round(p, 6)})
    sums = {}
    for row in probabilities:
        sums[row["place"]] = sums.get(row["place"], 0.0) + row["p"]
    for row in probabilities:
        if sums[row["place"]] > 1.0:
            over = sums[row["place"]] - 1.0
            row["p"] = round(max(row["p"] - over, 0.0), 6)
            sums[row["place"]] -= over
    mass = {}
    for (uid, pl), p in probs.items():
        mass[uid] = mass.get(uid, 0.0) + p * (places + 1 - pl)
    top10 = [uid for uid, _ in sorted(mass.items(), key=lambda kv: (-kv[1], kv[0]))[:places]]
    if len(top10) < places:
        seen = set(top10)
        for uid in sorted(last_chart, key=lambda u: last_chart[u]):
            if uid not in seen:
                top10.append(uid)
                seen.add(uid)
            if len(top10) == places:
                break
    return {"probabilities": probabilities, "top10": top10}, diag


def main():
    input_dir = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
    output_dir = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")
    task, reads = load_reads(input_dir)
    forecast, _ = build_forecast(task, reads)
    with open(os.path.join(output_dir, task["output"]), "w") as f:
        f.write(json.dumps(forecast) + "\n")


if __name__ == "__main__":
    main()
