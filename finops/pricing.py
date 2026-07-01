"""Pricing & purchasing economics — measure in $/1M-token, not $/GPU-hr.

Figures are June-2026 as-of snapshots from the deck's RESEARCH dossier; treat
live prices as fast-moving (re-baseline before each cohort).
"""
from __future__ import annotations

# ---- Extension 1: GPU-type interruption rates (Extension 1 — better tier policy) ----
# Real-world spot-interruption rates vary by GPU family: premium SKUs (H100/H200/B200)
# on neoclouds are typically reclaimed less often than commodity SKUs (L4/A10G).
# Per-hour probability of spot reclaim used to grade the spot-vs-reserved trade-off.
GPU_INTERRUPT_RATE = {
    "H100":  0.03,   # hot SKU, low reclaim
    "H200":  0.03,
    "B200":  0.04,
    "A100":  0.05,
    "A10G":  0.06,
    "L4":    0.06,
    "MI300X":0.05,
}
DEFAULT_INTERRUPT_RATE = 0.05

# Reserved-commitment minimum durations (days). Reserved < duration is wasted $ for the
# remainder of the contract.
RESERVED_MIN_DAYS = {"1yr": 365, "3yr": 1095}


def request_cost(
    input_tok: int,
    output_tok: int,
    price_in_per_m: float,
    price_out_per_m: float,
    cached_in: int = 0,
    cache_discount: float = 0.10,   # Anthropic cached-read ~0.1x (=-90%)
    batch: bool = False,
    batch_discount: float = 0.50,   # Batch API ~ -50%
) -> float:
    """USD cost of a single request. Cached input billed at cache_discount x price."""
    cached_in = min(max(0, cached_in), input_tok)
    uncached_in = input_tok - cached_in
    cost = (
        (uncached_in / 1e6) * price_in_per_m
        + (cached_in / 1e6) * price_in_per_m * cache_discount
        + (output_tok / 1e6) * price_out_per_m
    )
    if batch:
        cost *= batch_discount
    return cost


def dollars_per_million(total_cost_usd: float, total_tokens: int) -> float:
    """Aggregate unit economics: $ per 1,000,000 tokens served."""
    if total_tokens <= 0:
        return 0.0
    return total_cost_usd / (total_tokens / 1e6)


def discount_stack(
    batch: bool = False,
    cache_hit_frac: float = 0.0,
    batch_discount: float = 0.50,
    cache_discount: float = 0.10,
) -> float:
    """Effective fraction of the naive bill after stacking discounts (input-heavy view).

    Discounts MULTIPLY: cache applies to the cached share of input, batch to the
    whole bill. batch + 100% cache-hit -> 0.5 * 0.1 = 0.05 (~95% off).
    """
    cache_mult = cache_hit_frac * cache_discount + (1.0 - cache_hit_frac)
    batch_mult = batch_discount if batch else 1.0
    return cache_mult * batch_mult


def break_even_utilization(discount_frac: float) -> float:
    """Utilization at which a commitment pays off ~= 1 - discount.

    A 45% reserved discount needs ~55% utilization (~13.2h/day) to beat on-demand.
    """
    return max(0.0, min(1.0, 1.0 - discount_frac))


def recommend_tier(hours_per_day: float, interruptible: bool, reserved_discount: float = 0.45) -> str:
    """Pick a purchasing tier from a workload's duty cycle + interruptibility.

    DOCUMENTED simple policy (instructor extension point — swap in your own):
      - interruptible & not 24/7  -> 'spot'      (checkpoint and ride the discount)
      - duty cycle >= break-even  -> 'reserved'  (steady, high utilization)
      - otherwise                 -> 'on_demand' (spiky / low duty)
    """
    duty = max(0.0, hours_per_day) / 24.0
    be = break_even_utilization(reserved_discount)
    if interruptible and hours_per_day < 24:
        return "spot"
    if duty >= be:
        return "reserved"
    return "on_demand"


def interrupt_rate_for(gpu_type: str | None) -> float:
    """Per-hour spot interruption probability for a GPU family (Extension 1)."""
    if not gpu_type:
        return DEFAULT_INTERRUPT_RATE
    return GPU_INTERRUPT_RATE.get(gpu_type, DEFAULT_INTERRUPT_RATE)


def recommend_tier_v2(
    hours_per_day: float,
    interruptible: bool,
    gpu_type: str | None = None,
    job_days: int | None = None,
    on_demand_hr: float | None = None,
    spot_hr: float | None = None,
    reserved_1yr_hr: float | None = None,
    reserved_3yr_hr: float | None = None,
    spot_interrupt_threshold: float = 0.07,
) -> dict:
    """Improved tier policy (Extension 1).

    Improvements over `recommend_tier`:
      1. GPU-type-aware interruption rate: when interrupt_rate exceeds
         `spot_interrupt_threshold`, spot is no longer attractive — fall back
         to reserved (if duty allows) or on_demand.
      2. Reserved-term awareness: a workload shorter than the commitment window
         (365d for 1yr, 1095d for 3yr) cannot commit to that tier — wasted $ for
         the unused portion of the contract. Pick the longest affordable term.
      3. If price columns are passed, picks the *cheapest* reserved term that
         meets break-even (1yr vs 3yr) given the workload duration.

    Returns a dict with keys: tier, reason, duty, break_even, interrupt_rate,
    and (when prices supplied) per-term savings.
    """
    duty = max(0.0, float(hours_per_day)) / 24.0
    int_rate = interrupt_rate_for(gpu_type)
    be_3yr = break_even_utilization((1.0 - (reserved_3yr_hr / on_demand_hr))) \
        if (on_demand_hr and reserved_3yr_hr and on_demand_hr > 0) else break_even_utilization(0.45)
    be_1yr = break_even_utilization((1.0 - (reserved_1yr_hr / on_demand_hr))) \
        if (on_demand_hr and reserved_1yr_hr and on_demand_hr > 0) else break_even_utilization(0.20)

    decision = {"duty": round(duty, 3), "interrupt_rate": int_rate,
                "break_even_1yr": round(be_1yr, 3), "break_even_3yr": round(be_3yr, 3)}

    # --- Rule A: interruptibility + reclaim rate ---
    if interruptible and hours_per_day < 24 and int_rate < spot_interrupt_threshold:
        decision.update({"tier": "spot", "reason":
            f"interruptible, low reclaim ({int_rate:.0%}/h < {spot_interrupt_threshold:.0%}/h threshold)"})
        return decision
    if interruptible and int_rate >= spot_interrupt_threshold:
        decision["reason"] = (f"reclaim too high ({int_rate:.0%}/h >= {spot_interrupt_threshold:.0%}/h); "
                              "spot rework cost erodes the discount")

    # --- Rule B: reserved commitment duration vs job_days ---
    eligible_3yr = (job_days is None) or (job_days >= RESERVED_MIN_DAYS["3yr"])
    eligible_1yr = (job_days is None) or (job_days >= RESERVED_MIN_DAYS["1yr"])

    # --- Rule C: duty-cycle vs break-even for the best eligible reserved term ---
    chosen_reserved_term = None
    if eligible_3yr and reserved_3yr_hr is not None and on_demand_hr and duty >= be_3yr:
        chosen_reserved_term = "3yr"
        decision["reserved_term"] = "3yr"
    elif eligible_1yr and reserved_1yr_hr is not None and on_demand_hr and duty >= be_1yr:
        chosen_reserved_term = "1yr"
        decision["reserved_term"] = "1yr"

    if chosen_reserved_term:
        decision["tier"] = "reserved"
        decision["reason"] = f"duty {duty:.0%} >= break-even for {chosen_reserved_term} reserved"
        return decision

    decision["tier"] = "on_demand"
    if "reason" not in decision:
        if job_days is not None and job_days < RESERVED_MIN_DAYS["1yr"]:
            decision["reason"] = (f"job ({job_days}d) too short for any reserved commit; "
                                  "duty too low to amortize the risk")
        else:
            decision["reason"] = f"duty {duty:.0%} too low to break even on reserved"
    return decision


def cached_reads_for_break_even(
    write_cost_per_m: float,
    read_discount: float = 0.10,
    write_units_per_m: float = 1.0,
) -> float:
    """Minimum average cached reads to make prompt caching pay off (Extension 3).

    Caching is only profitable when the savings from read-discount on cached
    input exceed the write/storage cost. Returns the break-even number of
    re-reads of the same prefix. Below 1.0 the cache never pays off.

    write_cost_per_m  — $ per 1M tokens to write the cache prefix
    read_discount     — cached read multiplier (0.10 = 90% off)
    write_units_per_m — units of input paid as write cost (Anthropic charges
                        ~1.25x the input price to write a cache, Gemini charges
                        separately for storage; this lets callers calibrate).
    """
    if write_cost_per_m <= 0:
        return 1.0
    # Each re-read saves (1 - read_discount) of the prefix read cost; first read
    # is the write itself. Break-even when cumulative savings = write cost.
    # write_cost / ((1 - read_discount) * unit_cost) ~= write_units_per_m / (1 - read_discount).
    saving_per_read = max(1e-9, 1.0 - read_discount)
    return write_units_per_m / saving_per_read


def cache_is_worth_it(
    avg_cache_reads: float,
    write_cost_per_m: float = 3.75,   # ~1.25x Anthropic Sonnet input price
    read_discount: float = 0.10,
    write_units_per_m: float = 1.25,
) -> dict:
    """Decide whether prompt caching saves money for a given workload (Extension 3).

    Returns a dict: {'worth_it': bool, 'break_even_reads': float, 'avg_reads': float,
                     'explanation': str}.

    Cache only pays when the prefix is re-read enough times that the per-read
    savings exceed the one-time write cost. Below ~1.4 reads for Anthropic's
    1.25x write premium + 90% read discount, caching is a net loss.
    """
    be = cached_reads_for_break_even(write_cost_per_m, read_discount, write_units_per_m)
    worth = avg_cache_reads >= be
    msg = (f"avg re-reads {avg_cache_reads:.2f} >= break-even {be:.2f}; "
           f"cache pays off (net saving ~${(avg_cache_reads - be) * (1 - read_discount):.3f}/M-tok per read)"
           if worth else
           f"avg re-reads {avg_cache_reads:.2f} < break-even {be:.2f}; "
           f"cache loses ~${(be - avg_cache_reads) * (1 - read_discount):.3f}/M-tok per read")
    return {"worth_it": worth, "break_even_reads": round(be, 3),
            "avg_reads": round(avg_cache_reads, 3), "explanation": msg}


def spot_checkpoint_cost(
    job_hours: float,
    spot_hr: float,
    on_demand_hr: float,
    interrupt_rate: float = 0.05,      # per-hour chance (H100 spot ~<5%)
    ckpt_overhead_frac: float = 0.03,  # steady cost of writing checkpoints
    rework_hours_per_interrupt: float = 0.5,
) -> dict:
    """Effective cost of running a checkpointable job on spot vs on-demand.

    Interruptions waste the compute since the last checkpoint (rework); checkpointing
    adds a small steady overhead. Spot still wins for interruptible jobs.
    """
    expected_interrupts = job_hours * interrupt_rate
    rework_hours = expected_interrupts * rework_hours_per_interrupt
    effective_hours = job_hours * (1.0 + ckpt_overhead_frac) + rework_hours
    spot_cost = effective_hours * spot_hr
    on_demand_cost = job_hours * on_demand_hr
    savings_pct = (1.0 - spot_cost / on_demand_cost) * 100.0 if on_demand_cost > 0 else 0.0
    return {
        "spot_effective_hours": round(effective_hours, 2),
        "spot_cost": round(spot_cost, 2),
        "on_demand_cost": round(on_demand_cost, 2),
        "savings_pct": round(savings_pct, 1),
    }
