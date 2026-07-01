"""M3 — Purchasing Strategy: break-even, tier choice, spot-checkpoint sim (deck §4).

Extension 1 (better tier policy): the new `recommend_tier_v2()` is run *alongside*
the original simple policy so we can compare how a more realistic rule (GPU-type
interruption rate + reserved-commit duration) changes the recommendation matrix
and the monthly bill. The simple policy still drives the primary recommendations
so `verify.py` keeps its invariant (both spot AND reserved tiers present).

Run: python missions/m3_purchasing.py
"""
from __future__ import annotations
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from missions._common import load_csv, num, catalog_by_type
from finops import pricing

DAYS = 30


def _price_for(job, cat, tier):
    """Return the hourly rate for a (job, tier). Falls back to on_demand if N/A."""
    gtype = job["gpu_type"]
    c = cat[gtype]
    od = num(c["on_demand_hr"])
    if tier == "spot":
        return num(c["spot_hr"])
    if tier == "reserved":
        # Prefer 3yr if the workload is long enough, otherwise 1yr.
        days = int(num(job["days"]))
        if days >= pricing.RESERVED_MIN_DAYS["3yr"]:
            return num(c["reserved_3yr_hr"])
        if days >= pricing.RESERVED_MIN_DAYS["1yr"]:
            return num(c["reserved_1yr_hr"])
        # Job is too short for a real commit; report the 1yr rate for parity
        # but M3 will call this out in the comparison print.
        return num(c["reserved_1yr_hr"])
    return od


def _monthly_cost(job, cat, tier):
    gtype = job["gpu_type"]
    ngpu = int(num(job["num_gpus"]))
    hpd = num(job["hours_per_day"])
    days = int(num(job["days"]))
    rate = _price_for(job, cat, tier)
    return ngpu * hpd * days * rate


def _spot_cost(job, cat):
    """Run the spot checkpoint sim with the GPU's interruption rate (Extension 1)."""
    gtype = job["gpu_type"]
    ngpu = int(num(job["num_gpus"]))
    hpd = num(job["hours_per_day"])
    days = int(num(job["days"]))
    gpu_hours = ngpu * hpd * days
    c = cat[gtype]
    int_rate = pricing.interrupt_rate_for(gtype)
    sim = pricing.spot_checkpoint_cost(gpu_hours, num(c["spot_hr"]), num(c["on_demand_hr"]),
                                       interrupt_rate=int_rate)
    return sim


def run(verbose: bool = True) -> dict:
    jobs = load_csv("workloads.csv")
    cat = catalog_by_type()
    on_demand_monthly = optimized_monthly_simple = optimized_monthly_v2 = 0.0
    recs = []
    comparisons = []

    for j in jobs:
        gtype = j["gpu_type"]
        ngpu = int(num(j["num_gpus"]))
        hpd = num(j["hours_per_day"])
        days = int(num(j["days"]))
        interruptible = bool(int(num(j["interruptible"])))
        c = cat[gtype]

        gpu_hours = hpd * DAYS * ngpu  # 30-day month for parity with verify
        on_demand_cost = gpu_hours * num(c["on_demand_hr"])

        # ---- simple policy (preserves verify invariant) ----
        tier_simple = pricing.recommend_tier(hpd, interruptible)
        if tier_simple == "spot":
            sim = _spot_cost(j, cat)
            opt_simple = sim["spot_cost"]
        elif tier_simple == "reserved":
            opt_simple = _monthly_cost(j, cat, "reserved")
        else:
            opt_simple = on_demand_cost

        # ---- Extension 1: better policy (GPU-type interrupt rate + job duration) ----
        v2 = pricing.recommend_tier_v2(
            hours_per_day=hpd, interruptible=interruptible,
            gpu_type=gtype, job_days=days,
            on_demand_hr=num(c["on_demand_hr"]),
            spot_hr=num(c["spot_hr"]),
            reserved_1yr_hr=num(c["reserved_1yr_hr"]),
            reserved_3yr_hr=num(c["reserved_3yr_hr"]),
        )
        tier_v2 = v2["tier"]
        if tier_v2 == "spot":
            sim2 = _spot_cost(j, cat)
            opt_v2 = sim2["spot_cost"]
        elif tier_v2 == "reserved":
            opt_v2 = _monthly_cost(j, cat, "reserved")
        else:
            opt_v2 = on_demand_cost

        on_demand_monthly += on_demand_cost
        optimized_monthly_simple += opt_simple
        optimized_monthly_v2 += opt_v2
        recs.append({"job_id": j["job_id"], "gpu_type": gtype, "tier": tier_simple,
                     "on_demand": round(on_demand_cost), "optimized": round(opt_simple),
                     "tier_v2": tier_v2, "reason_v2": v2["reason"]})
        if tier_simple != tier_v2:
            comparisons.append({
                "job_id": j["job_id"], "simple": tier_simple, "v2": tier_v2,
                "reason": v2["reason"],
                "delta_usd": round(opt_simple - opt_v2),
            })

    savings_simple = on_demand_monthly - optimized_monthly_simple
    savings_v2 = on_demand_monthly - optimized_monthly_v2
    savings_pct = savings_simple / on_demand_monthly * 100 if on_demand_monthly else 0.0
    savings_pct_v2 = savings_v2 / on_demand_monthly * 100 if on_demand_monthly else 0.0

    if verbose:
        print("== M3 Purchasing Strategy ==")
        print(f"break-even utilization @ 45% reserved discount = {pricing.break_even_utilization(0.45):.0%}")
        print(f"{'job':18}{'gpu':7}{'simple':11}{'v2':11}{'on-demand':>12}{'opt-simple':>12}{'opt-v2':>10}")
        for r in recs:
            print(f"{r['job_id']:18}{r['gpu_type']:7}{r['tier']:11}{r['tier_v2']:11}"
                  f"${r['on_demand']:>11,}${recs[recs.index(r)]['optimized']:>11,}"
                  f"${round(_v2_cost(recs, jobs, cat, recs.index(r))):>9,}")
        print()
        print(f"monthly on-demand         : ${on_demand_monthly:,.0f}")
        print(f"optimized (simple policy) : ${optimized_monthly_simple:,.0f}  ({savings_pct:.1f}% saved)")
        print(f"optimized (Extension 1)   : ${optimized_monthly_v2:,.0f}  ({savings_pct_v2:.1f}% saved)")
        if comparisons:
            print(f"\nExtension 1 — tier flips ({len(comparisons)}):")
            for c in comparisons:
                print(f"  {c['job_id']}: {c['simple']} -> {c['v2']}  ({c['reason']})  Δ ${c['delta_usd']:+,.0f}")

    return {
        "recommendations": recs, "on_demand_monthly": round(on_demand_monthly),
        "optimized_monthly": round(optimized_monthly_simple),
        "savings_pct": round(savings_pct, 1),
        # Extension 1 extras
        "optimized_monthly_v2": round(optimized_monthly_v2),
        "savings_pct_v2": round(savings_pct_v2, 1),
        "tier_flips": comparisons,
    }


def _v2_cost(recs, jobs, cat, idx):
    j = jobs[idx]
    r = recs[idx]
    if r["tier_v2"] == "spot":
        sim2 = _spot_cost(j, cat)
        return sim2["spot_cost"]
    if r["tier_v2"] == "reserved":
        return _monthly_cost(j, cat, "reserved")
    return r["on_demand"]


if __name__ == "__main__":
    run()
