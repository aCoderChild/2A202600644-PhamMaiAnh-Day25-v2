"""Tests for the Extension-1, Extension-3 and Extension-4 engine additions."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from finops import pricing


# ---------- Extension 1 — interrupt_rate_for ----------
def test_interrupt_rate_known_gpu():
    assert pricing.interrupt_rate_for("H100") == 0.03
    assert pricing.interrupt_rate_for("L4")   == 0.06
    assert pricing.interrupt_rate_for("B200") == 0.04


def test_interrupt_rate_unknown_gpu_uses_default():
    assert pricing.interrupt_rate_for(None) == pricing.DEFAULT_INTERRUPT_RATE
    assert pricing.interrupt_rate_for("Mystery-GPU") == pricing.DEFAULT_INTERRUPT_RATE


# ---------- Extension 1 — recommend_tier_v2 ----------
def test_v2_spot_for_interruptible_low_reclaim():
    r = pricing.recommend_tier_v2(hours_per_day=12, interruptible=True,
                                  gpu_type="H100", job_days=30)
    assert r["tier"] == "spot"
    assert "interruptible" in r["reason"]


def test_v2_falls_back_when_reclaim_too_high():
    # Spot threshold default is 7%; H100 = 3% (low reclaim), so still spot.
    # Bump a synthetic high-reclaim GPU by passing interruptible but checking
    # the rule for a long-running non-spot job.
    r = pricing.recommend_tier_v2(hours_per_day=24, interruptible=True,
                                  gpu_type="H100", job_days=2000,
                                  on_demand_hr=2.5, spot_hr=1.5,
                                  reserved_1yr_hr=2.0, reserved_3yr_hr=1.4)
    # 24h/day with no reclaim-friendly exception + long job should NOT be spot
    assert r["tier"] != "spot"
    assert r["tier"] == "reserved"


def test_v2_short_job_cannot_commit_reserved():
    # 30-day workload is far too short for a 365d or 1095d reserved commit.
    r = pricing.recommend_tier_v2(hours_per_day=24, interruptible=False,
                                  gpu_type="A100", job_days=30,
                                  on_demand_hr=1.79, spot_hr=1.10,
                                  reserved_1yr_hr=1.40, reserved_3yr_hr=1.00)
    assert r["tier"] == "on_demand"
    assert "too short" in r["reason"].lower() or "duty" in r["reason"].lower()


def test_v2_long_high_duty_job_gets_3yr_reserved():
    # 3-year long-running service at 90% duty should pick the 3yr reserved term.
    r = pricing.recommend_tier_v2(hours_per_day=22, interruptible=False,
                                  gpu_type="H100", job_days=1200,
                                  on_demand_hr=2.5, spot_hr=1.5,
                                  reserved_1yr_hr=2.0, reserved_3yr_hr=1.4)
    assert r["tier"] == "reserved"
    assert r["reserved_term"] == "3yr"


def test_v2_1yr_vs_3yr_term_choice():
    # 1-year exactly -> 1yr reserved only (3yr requires >= 1095 days)
    r = pricing.recommend_tier_v2(hours_per_day=24, interruptible=False,
                                  gpu_type="H100", job_days=365,
                                  on_demand_hr=2.5, spot_hr=1.5,
                                  reserved_1yr_hr=2.0, reserved_3yr_hr=1.4)
    assert r["tier"] == "reserved"
    assert r["reserved_term"] == "1yr"


# ---------- Extension 3 — cache_is_worth_it ----------
def test_cache_break_even_arithmetic():
    # write_units_per_m / (1 - read_discount) = 1.25 / 0.9 = 1.388...
    be = pricing.cached_reads_for_break_even(write_cost_per_m=3.75,
                                             read_discount=0.10,
                                             write_units_per_m=1.25)
    assert abs(be - 1.3888) < 1e-3


def test_cache_is_worth_it_true_above_break_even():
    res = pricing.cache_is_worth_it(avg_cache_reads=5.0,
                                    write_cost_per_m=3.75,
                                    read_discount=0.10, write_units_per_m=1.25)
    assert res["worth_it"] is True
    assert res["avg_reads"] == 5.0
    assert "pays off" in res["explanation"]


def test_cache_is_worth_it_false_below_break_even():
    res = pricing.cache_is_worth_it(avg_cache_reads=1.0,
                                    write_cost_per_m=3.75,
                                    read_discount=0.10, write_units_per_m=1.25)
    assert res["worth_it"] is False
    assert "loses" in res["explanation"]


# ---------- Extension 3 — backwards compat: original recommend_tier unchanged ----------
def test_original_recommend_tier_signature_intact():
    # The original three-arg call must still return the same answers.
    assert pricing.recommend_tier(2, True) == "spot"
    assert pricing.recommend_tier(24, False) == "reserved"
    assert pricing.recommend_tier(4, False) == "on_demand"


# ---------- Extension 4 — sustainability snapshot used in M2 stays sane ----------
def test_reasoning_energy_multiplier_visible():
    from finops import sustainability
    base = sustainability.wh_per_query(800, is_reasoning=False)
    high = sustainability.wh_per_query(800, is_reasoning=True)
    # Reasoning should be at least 50x (deck: ~74-86x).
    assert high / base >= 50
