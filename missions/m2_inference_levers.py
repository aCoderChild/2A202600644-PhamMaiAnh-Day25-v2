"""M2 — Inference Cost Levers: $/1M-token, batch x cache x cascade (deck §7).

Extensions:
  * Extension 3 — `cache_is_worth_it()` gates the cache discount so we never
    claim savings the cache cannot actually deliver.
  * Extension 4 — reasoning traffic ($ + Wh) is reported separately so a
    FinOps lead can see how much the reasoning tier is actually costing.

Run: python missions/m2_inference_levers.py
"""
from __future__ import annotations
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from collections import defaultdict
from missions._common import load_csv, num
from finops import pricing, sustainability

# $/1M tokens (input, output) — illustrative 2026.
MODEL_PRICES = {"small": (0.20, 0.40), "large": (3.00, 15.00)}
# Per-million write cost to seed a prompt cache (illustrative — Anthropic ~1.25x
# input price; Gemini charges separately for storage).
CACHE_WRITE_COST = {"small": 0.25, "large": 3.75}   # $/M tokens written to cache
CACHE_READ_DISCOUNT = 0.10                          # 90% off the read side


def _avg_cache_reads(rows: list[dict]) -> float:
    """Empirical estimate of average re-reads per cacheable prefix (Extension 3).

    Heuristic: requests sharing the same (team, hour) bucket re-use the same
    system-prompt prefix; re-reads ~= #requests with cached>0 in the bucket.
    Total re-reads / #buckets ~= avg_cache_reads across the workload.
    """
    bucket_hits: dict[tuple[str, str], int] = defaultdict(int)
    for r in rows:
        if int(num(r["cached_input_tokens"])) > 0:
            bucket_hits[(r["team"], r["ts"][:13])] += 1
    if not bucket_hits:
        return 1.0
    total_reads = sum(bucket_hits.values())
    n_unique_prefixes = len(bucket_hits)
    return total_reads / n_unique_prefixes


def run(verbose: bool = True) -> dict:
    rows = load_csv("token_usage.csv")
    base_cost = opt_cost = 0.0
    base_cost_reasoning = opt_cost_reasoning = 0.0
    base_cost_normal = opt_cost_normal = 0.0
    wh_reasoning = wh_normal = 0.0
    total_tokens = total_reasoning_tokens = total_normal_tokens = 0
    total_cached_tokens = 0

    # ---- Extension 3: empirically measure avg_cache_reads and gate the cache ----
    avg_reads = _avg_cache_reads(rows)
    # Tier-specific write cost: large-model prefixes are more expensive to write,
    # so break-even reads differ per tier.
    be_small = pricing.cached_reads_for_break_even(
        CACHE_WRITE_COST["small"], CACHE_READ_DISCOUNT, write_units_per_m=1.25)
    be_large = pricing.cached_reads_for_break_even(
        CACHE_WRITE_COST["large"], CACHE_READ_DISCOUNT, write_units_per_m=1.25)
    worth_small = pricing.cache_is_worth_it(avg_reads, CACHE_WRITE_COST["small"],
                                            CACHE_READ_DISCOUNT, 1.25)["worth_it"]
    worth_large = pricing.cache_is_worth_it(avg_reads, CACHE_WRITE_COST["large"],
                                            CACHE_READ_DISCOUNT, 1.25)["worth_it"]

    for r in rows:
        inp, out = int(num(r["input_tokens"])), int(num(r["output_tokens"]))
        cached = int(num(r["cached_input_tokens"]))
        is_batch = bool(int(num(r["is_batch"])))
        is_reasoning = bool(int(num(r["is_reasoning"])))
        total_tokens += inp + out
        total_cached_tokens += cached
        if is_reasoning:
            total_reasoning_tokens += inp + out
        else:
            total_normal_tokens += inp + out

        # BASELINE: everything on the large model, no cache, no batch
        lin, lout = MODEL_PRICES["large"]
        b = pricing.request_cost(inp, out, lin, lout)
        base_cost += b
        (base_cost_reasoning if is_reasoning else base_cost_normal).__iadd__(b) if False else None
        if is_reasoning:
            base_cost_reasoning += b
        else:
            base_cost_normal += b

        # OPTIMIZED: cascade + batch. Cache is gated per tier (Extension 3).
        pin, pout = MODEL_PRICES[r["route_tier"]]
        effective_cache = 0
        if r["route_tier"] == "small" and worth_small:
            effective_cache = cached
        elif r["route_tier"] == "large" and worth_large:
            effective_cache = cached
        o = pricing.request_cost(inp, out, pin, pout,
                                 cached_in=effective_cache, batch=is_batch)
        opt_cost += o
        if is_reasoning:
            opt_cost_reasoning += o
            wh_reasoning += sustainability.wh_per_query(inp + out, is_reasoning=True)
        else:
            opt_cost_normal += o
            wh_normal += sustainability.wh_per_query(inp + out, is_reasoning=False)

    base_pm = pricing.dollars_per_million(base_cost, total_tokens)
    opt_pm = pricing.dollars_per_million(opt_cost, total_tokens)
    savings_pct = (1 - opt_cost / base_cost) * 100 if base_cost else 0.0

    # ---- Extension 4: reasoning-share of cost and energy ----
    reasoning_share_pct = (base_cost_reasoning / base_cost * 100) if base_cost else 0.0
    reasoning_traffic_pct = (total_reasoning_tokens / total_tokens * 100) if total_tokens else 0.0
    wh_per_query_reasoning = (wh_reasoning / max(1, sum(1 for r in rows if int(num(r["is_reasoning"])))))
    wh_per_query_normal = (wh_normal / max(1, sum(1 for r in rows if not int(num(r["is_reasoning"])))))

    if verbose:
        print("== M2 Inference Cost Levers ==")
        print(f"requests={len(rows)}  tokens={total_tokens:,}  cached={total_cached_tokens:,}")
        print(f"baseline  : ${base_cost:,.2f}/day   ${base_pm:.3f}/1M-token")
        print(f"optimized : ${opt_cost:,.2f}/day   ${opt_pm:.3f}/1M-token")
        print(f"savings   : {savings_pct:.1f}%  (cascade + caching + batch)")
        print(f"discount stack (batch + 100% cache): {pricing.discount_stack(batch=True, cache_hit_frac=1.0):.3f} of naive")
        print()
        print("Extension 3 — cache economics (per-million write cost):")
        print(f"  empirical avg re-reads / prefix: {avg_reads:.2f}")
        print(f"  break-even reads  small={be_small:.2f}  large={be_large:.2f}")
        print(f"  cache worth it?  small={worth_small}  large={worth_large}")
        print("Extension 4 — reasoning budget:")
        print(f"  traffic share: {reasoning_traffic_pct:.1f}% of tokens ({total_reasoning_tokens:,} tok)")
        print(f"  cost   share: {reasoning_share_pct:.1f}% of $    (${base_cost_reasoning:,.2f}/day)")
        print(f"  Wh/query     : reasoning {wh_per_query_reasoning:,.2f}  vs normal {wh_per_query_normal:.4f}  "
              f"({wh_per_query_reasoning / max(1e-9, wh_per_query_normal):.1f}x)")

    return {
        "baseline_daily": round(base_cost, 2), "optimized_daily": round(opt_cost, 2),
        "baseline_per_m": round(base_pm, 3), "optimized_per_m": round(opt_pm, 3),
        "savings_pct": round(savings_pct, 1), "total_tokens": total_tokens,
        # Extension 3
        "avg_cache_reads": round(avg_reads, 3),
        "cache_break_even_small": round(be_small, 3),
        "cache_break_even_large": round(be_large, 3),
        "cache_worth_small": worth_small, "cache_worth_large": worth_large,
        # Extension 4
        "reasoning_traffic_pct": round(reasoning_traffic_pct, 2),
        "reasoning_cost_pct": round(reasoning_share_pct, 2),
        "reasoning_cost_daily": round(base_cost_reasoning, 2),
        "reasoning_wh_query": round(wh_per_query_reasoning, 3),
        "normal_wh_query": round(wh_per_query_normal, 4),
    }


if __name__ == "__main__":
    run()
