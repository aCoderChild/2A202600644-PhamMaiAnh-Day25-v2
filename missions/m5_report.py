"""M5 — Báo cáo Tối ưu hóa: tổng hợp M1-M4 thành baseline-vs-optimized (deck §1/§11).

Sinh báo cáo markdown đa mục bằng **tiếng Việt**:
  - Tổng quan (baseline / tối ưu / % tiết kiệm / $/1M-token)
  - Lời nói dối của GPU-Util (phát hiện M1 + tác động tài chính)
  - Tiết kiệm theo đòn bẩy
  - Kết quả phần mở rộng (cache economics, reasoning budget, tier policy v2)
  - Tính bền vững (năng lượng, carbon, đánh đổi vùng, năng lượng reasoning)
  - Khuyến nghị ưu tiên cho NimbusAI

Chạy: python missions/m5_report.py   ->  outputs/report.md + outputs/savings.png
"""
from __future__ import annotations
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import os
from missions._common import num, catalog_by_type, ROOT
from finops import report, sustainability, pricing
from missions import m1_efficiency_audit, m2_inference_levers, m3_purchasing

DAYS = 30
# Một tier xuống cho các GPU bị "nói dối" (util cao nhưng MFU thấp)
RIGHTSIZE_MAP = {"H100": "A100", "H200": "H100", "A100": "A10G", "A10G": "L4", "L4": "L4"}


def _format_usd(x: float) -> str:
    return f"${x:,.0f}"


# Ánh xạ câu "reason" tiếng Anh từ `recommend_tier_v2()` (pricing.py) sang tiếng Việt
# để báo cáo hiển thị thuần nhất. Không sửa engine — chỉ là lớp trình bày (UI layer).
_REASON_VI = {
    "interruptible, low reclaim": "interruptible, tỷ lệ reclaim thấp",
    "interruptible, low reclaim (3% / h < 7% / h threshold)":
        "interruptible, tỷ lệ reclaim thấp (3%/h < 7%/h ngưỡng)",
    "reclaim too high": "tỷ lệ reclaim quá cao",
    "duty 100% >= break-even for 3yr reserved":
        "duty 100% >= break-even cho reserved 3 năm",
    "duty 75% >= break-even for 1yr reserved":
        "duty 75% >= break-even cho reserved 1 năm",
    "job (30d) too short for any reserved commit; duty too low to amortize the risk":
        "job (30 ngày) quá ngắn cho bất kỳ cam kết reserved nào; duty không đủ cao để khấu hao rủi ro",
}


def _vi_reason(reason: str) -> str:
    """Dịch câu lý do từ `recommend_tier_v2()` sang tiếng Việt (không bắt buộc khớp tuyệt đối)."""
    if not reason:
        return ""
    for en, vi in _REASON_VI.items():
        if en in reason:
            return reason.replace(en, vi)
    return reason


def _build_headline(baseline, optimized, levers) -> str:
    """Phần tổng quan đầu báo cáo — viết thẳng tiếng Việt."""
    savings = baseline - optimized
    pct = (savings / baseline * 100.0) if baseline > 0 else 0.0
    lines = [
        "# NimbusAI — Báo cáo Tối ưu hóa Chi phí GPU",
        "",
        f"**Kỳ báo cáo:** hàng tháng  ",
        f"**Chi phí baseline:** ${baseline:,.0f}  ",
        f"**Chi phí tối ưu:** ${optimized:,.0f}  ",
        f"**Tiết kiệm dự kiến:** ${savings:,.0f}  (**{pct:.0f}%**)",
        "",
        "## Tiết kiệm theo đòn bẩy",
        "",
        "| Đòn bẩy | Tiết kiệm (USD) |",
        "|---|---|",
    ]
    for name, amount in levers.items():
        lines.append(f"| {name} | ${amount:,.0f} |")
    return "\n".join(lines)


def _build_extra_sections(r1, r2, r3) -> list:
    """Các mục phân tích sâu — tiếng Việt."""
    cat = catalog_by_type()
    lines: list[str] = []

    # --- Lời nói dối của GPU-Util ---
    lines += ["", "## Lời nói dối của GPU-Util", "",
              "**`nvidia-smi` chỉ báo `gpu_util_pct` — đó là tín hiệu *clock-busy* (có xung "
              "nhịp chạy), KHÔNG phải hiệu quả tính toán.** Một GPU có thể ở 98% util "
              "mà chỉ đạt 20% FLOPs đỉnh khi memory stall, kernel-launch overhead, hoặc "
              "I/O wait chiếm ưu thế trong step. **Bạn trả tiền cho cả giờ H100 nhưng "
              "chỉ nhận được 1/5 FLOPs.**",
              ""]
    if r1["lies"]:
        lie_lines = ["| GPU | Loại | Util% | MFU | Idle (giờ) |",
                     "|---|---|---|---|---|"]
        for lie in r1["lies"]:
            lie_lines.append(f"| {lie['gpu_id']} | {lie['gpu_type']} | "
                             f"{lie['gpu_util_pct']:.0f} | {lie['mfu']:.2f} | "
                             f"{lie['idle_hours']} |")
        lines += lie_lines + ["",
                              "**Tác động tài chính:** các GPU này đang chạy H100 nhưng "
                              "chỉ cho ra compute hạng A. Hạ cấp xuống một tier bên dưới và "
                              "tắt các giờ thực sự idle sẽ thu hồi lại phần chi phí đã lãng phí "
                              "mà không làm giảm throughput."]
    else:
        lines.append("Không phát hiện GPU-Util lie nào trong cửa sổ quan sát này.")

    # --- Tính bền vững ---
    median_tokens = 800
    wh = sustainability.wh_per_query(median_tokens)
    carbon_us = sustainability.carbon_g(wh, "us-east-1")
    carbon_no = sustainability.carbon_g(wh, "europe-north1")
    carbon_pl = sustainability.carbon_g(wh, "europe-central2")
    energy_us = sustainability.energy_cost_usd(wh, "us-east-1")
    energy_no = sustainability.energy_cost_usd(wh, "europe-north1")
    carbon_cleanest_pct = (1 - carbon_no / carbon_us) * 100 if carbon_us > 0 else 0
    energy_cheap_pct = (1 - energy_no / energy_us) * 100 if energy_us > 0 else 0
    lines += ["", "## Tính bền vững", "",
              "**Chi phí mỗi truy vấn (trung vị ~800 tok):**", "",
              f"- Năng lượng / truy vấn: **{wh:.2f} Wh**",
              f"- Carbon / truy vấn (us-east-1, baseline): **{carbon_us:.3f} gCO2e**",
              f"- Carbon / truy vấn (europe-north1, sạch nhất): **{carbon_no:.3f} gCO2e**",
              f"- Carbon / truy vấn (europe-central2, dơ nhất): **{carbon_pl:.3f} gCO2e**",
              f"- Chi phí điện / truy vấn: us-east-1 **${energy_us:.5f}**  vs  "
              f"europe-north1 **${energy_no:.5f}**",
              "",
              "**Đánh đổi vùng triển khai:** europe-north1 (Na Uy, chủ yếu thủy điện) "
              f"vừa **{carbon_us / max(1e-9, carbon_no):.0f}× sạch hơn** vừa "
              f"**{energy_cheap_pct:.0f}% rẻ hơn** us-east-1 tính trên mỗi kWh. Chuyển "
              "các workload training có thể gián đoạn sang vùng này là động tác carbon "
              "rẻ nhất trên bàn cờ. europe-central2 (Ba Lan, ~660 g/kWh) là tệ nhất — "
              "tránh triển khai mới.",
              ""]

    # Reasoning-energy callout (Extension 4)
    ratio = 0.0
    if r2.get("reasoning_wh_query") and r2.get("normal_wh_query"):
        ratio = r2["reasoning_wh_query"] / max(1e-9, r2["normal_wh_query"])
        lines += ["**Tầng reasoning (Extension 4):**", "",
                  f"- Tỷ trọng traffic: **{r2['reasoning_traffic_pct']:.1f}%** token  |  "
                  f"Tỷ trọng chi phí: **{r2['reasoning_cost_pct']:.1f}%** $ "
                  f"(${r2['reasoning_cost_daily']:.2f}/ngày)",
                  f"- Năng lượng / truy vấn: reasoning **{r2['reasoning_wh_query']:.2f} Wh** "
                  f"so với thường **{r2['normal_wh_query']:.4f} Wh** "
                  f"→ reasoning nặng gấp **{ratio:.0f}×**.",
                  "- Quy tắc routing: giới hạn `is_reasoning=1` xuống <10% traffic; "
                  "đẩy phần đuôi dài các prompt \"hơi phức tạp\" sang tier rẻ hơn khi "
                  "bộ phân loại độ tin cậy nói rằng task không cần chain-of-thought."]

    # --- Chi tiết từng đòn bẩy ---
    lines += ["", "## Chi tiết từng đòn bẩy", ""]

    # Inference (Extension 3 + 4)
    lines += ["### Inference (cascade + cache + batch) — Extension 3 + 4", "",
              f"- Baseline: **${r2['baseline_per_m']:.3f} / 1M-token**  →  "
              f"Tối ưu: **${r2['optimized_per_m']:.3f} / 1M-token** "
              f"(**{r2['savings_pct']:.1f}%** giảm)",
              "- Ba đòn bẩy chồng lên nhau: cascade (route_tier), prompt caching, Batch API.",
              f"- Discount stack ở cache hit 100% + batch: "
              f"`{pricing.discount_stack(batch=True, cache_hit_frac=1.0):.3f}` của hóa đơn ngây thơ.",
              ""]
    if "avg_cache_reads" in r2:
        lines += [f"- **Extension 3 — kinh tế học cache:** số lần đọc lại trung bình / prefix "
                  f"= **{r2['avg_cache_reads']:.2f}**; break-even reads "
                  f"(small={r2['cache_break_even_small']:.2f}, "
                  f"large={r2['cache_break_even_large']:.2f}); "
                  f"cache có lợi cho cả hai tier (worth_small={r2['cache_worth_small']}, "
                  f"worth_large={r2['cache_worth_large']}).",
                  ""]

    # Purchasing (Extension 1)
    lines += ["### Mua GPU (spot / reserved) — Extension 1", "",
              f"- Savings với policy cũ: **{r3['savings_pct']:.1f}%** "
              f"(**{_format_usd(r3['on_demand_monthly'] - r3['optimized_monthly'])} / tháng**)",
              f"- Savings với policy Extension 1: **{r3['savings_pct_v2']:.1f}%** "
              f"(**{_format_usd(r3['on_demand_monthly'] - r3['optimized_monthly_v2'])} / tháng**)",
              ""]
    if r3["tier_flips"]:
        lines += ["- **Tier flips dưới policy thực tế hơn:**"]
        for flip in r3["tier_flips"]:
            lines.append(f"  - `{flip['job_id']}`: {flip['simple']} → {flip['v2']} "
                         f"({_vi_reason(flip['reason'])}; Δ ${flip['delta_usd']:+,.0f})")
        lines.append("")
        lines.append("  Insight: policy cũ đặt `reserved` cho mọi workload duty cao, "
                     "nhưng `reserved` yêu cầu cam kết 1–3 năm. Workload ngắn (một quý) "
                     "*tốt hơn* nên đi on-demand kể cả khi duty cycle cao — phần cam kết "
                     "không dùng hết còn nặng hơn khoản giảm giá theo giờ. Policy v2 cũng "
                     "cân nhắc interrupt rate theo loại GPU, nên các SKU hay bị reclaim "
                     "(A10G/L4) không bị đẩy mù quáng sang spot.")
    else:
        lines += ["- Policy v2 không làm thay đổi tier nào trong workload mix này."]

    # Right-size + idle
    rightsize_savings = 0.0
    for lie in r1["lies"]:
        cur = lie["gpu_type"]; tgt = RIGHTSIZE_MAP.get(cur, cur)
        rightsize_savings += max(0.0,
                                 num(cat[cur]["on_demand_hr"]) - num(cat[tgt]["on_demand_hr"])) * 24 * DAYS
    lines += ["", "### Right-size util-lies + Tắt GPU idle (M1)",
              "",
              f"- **Right-size util-lies**: hạ các GPU có MFU≈0.2 (H100/A100) xuống một "
              f"tier → thu hồi **${rightsize_savings:,.0f}/tháng** mà không mất throughput.",
              f"- **Tắt GPU idle**: idle_h × on_demand = **${r1['idle_waste_daily']:.2f}/ngày** "
              f"(quy ra tháng = ${r1['idle_waste_daily']*DAYS:,.0f}).",
              "",
              "_Mọi khoản tiết kiệm theo đòn bẩy đều đã nằm trong bảng tổng hợp ở đầu báo cáo._"]

    # --- Khuyến nghị ưu tiên ---
    lines += ["", "## Khuyến nghị ưu tiên", "",
              "1. **Cache mạnh trên assistant/rag** — số lần đọc lại thực nghiệm (~25) "
              "gấp ~20× ngưỡng break-even; chiết khấu 90% của cache là tiền thật.",
              "2. **Giới hạn reasoning <10% traffic** — reasoning nặng gấp "
              f"~{ratio:.0f}× năng lượng mỗi query; chỉ bật CoT khi bộ phân loại yêu cầu "
              "là đòn bẩy $/Wh lớn nhất.",
              "3. **Áp dụng policy tier v2** — workload ngắn ưu tiên on-demand hơn reserved; "
              "SKU hay reclaim ưu tiên on-demand hơn spot (chi phí rework > chiết khấu theo giờ).",
              "4. **Chuyển training có thể gián đoạn sang europe-north1** — cùng dải $/kWh, "
              "carbon thấp hơn ~13×. Kết hợp với spot + checkpoint, đây là vị trí vừa rẻ "
              "vừa sạch nhất cho batch job.",
              "5. **Tag đầy đủ trước khi chargeback** — coverage hiện ~92% (đóng nốt phần "
              "thiếu trước khi chuyển từ showback sang chargeback, để tránh team bị tính "
              "phí trên dữ liệu nhiễu).",
              "6. **Re-baseline hàng tháng** — mọi con số $/hr và $/kWh trong báo cáo là "
              "snapshot tháng 6/2026; các quy tắc cấu trúc (đo $/token, MFU thay GPU-Util, "
              "chồng chiết khấu, break-even trước khi cam kết reserved, tag trước chargeback) "
              "là bền vững — còn giá thì không."]

    lines += ["", "_Số liệu là snapshot tháng 6/2026; cần re-baseline trước khi áp dụng thực tế._"]
    return lines


def run(verbose: bool = True) -> dict:
    r1 = m1_efficiency_audit.run(verbose=False)
    r2 = m2_inference_levers.run(verbose=False)
    r3 = m3_purchasing.run(verbose=False)
    _ = None  # avoid unused warning; M4 được verify.py gọi riêng
    cat = catalog_by_type()

    # --- Phân bổ các khoản tiết kiệm ---
    infer_savings = (r2["baseline_daily"] - r2["optimized_daily"]) * DAYS
    purchasing_savings = r3["on_demand_monthly"] - r3["optimized_monthly"]
    idle_savings = r1["idle_waste_daily"] * DAYS
    rightsize_savings = 0.0
    for lie in r1["lies"]:
        cur = lie["gpu_type"]; tgt = RIGHTSIZE_MAP.get(cur, cur)
        rightsize_savings += max(0.0,
                                 num(cat[cur]["on_demand_hr"]) - num(cat[tgt]["on_demand_hr"])) * 24 * DAYS

    # Nhãn tiếng Việt cho cả bảng và biểu đồ waterfall
    levers = {
        "Inference (cascade/cache/batch)": round(infer_savings),
        "Mua GPU (spot/reserved)": round(purchasing_savings),
        "Right-size GPU 'nói dối'": round(rightsize_savings),
        "Tắt GPU idle": round(idle_savings),
    }
    baseline = r2["baseline_daily"] * DAYS + r3["on_demand_monthly"]
    optimized = baseline - sum(levers.values())
    total_pct = sum(levers.values()) / baseline * 100 if baseline else 0.0

    # --- Snapshot bền vững ---
    median_tokens = 800
    wh = sustainability.wh_per_query(median_tokens)
    sust = {
        "wh_per_query": wh,
        "carbon_g": sustainability.carbon_g(wh, "us-east-1"),
        "best_region": min(sustainability.REGION_CARBON, key=sustainability.REGION_CARBON.get),
    }

    # Báo cáo hoàn toàn bằng tiếng Việt — không dùng `report.build_report()` (giữ hàm đó
    # cho các caller khác và để test_build_report_has_savings còn pass).
    head_md = _build_headline(baseline, optimized, levers)
    extra = "\n".join(_build_extra_sections(r1, r2, r3))
    md = head_md + "\n" + extra

    out_md = os.path.join(ROOT, "outputs", "report.md")
    os.makedirs(os.path.dirname(out_md), exist_ok=True)
    with open(out_md, "w") as f:
        f.write(md)
    png = report.savings_waterfall(levers, os.path.join(ROOT, "outputs", "savings.png"))

    if verbose:
        print("== M5 Báo cáo Tối ưu hóa ==")
        print(md)
        print(f"\nĐã ghi: outputs/report.md" + (f" + outputs/savings.png" if png else " (matplotlib thiếu: bỏ qua PNG)"))

    return {"baseline_monthly": round(baseline), "optimized_monthly": round(optimized),
            "levers": levers, "total_savings_pct": round(total_pct, 1)}


if __name__ == "__main__":
    run()
