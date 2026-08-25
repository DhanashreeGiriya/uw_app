"""
Deterministic business-rules engine.

This intentionally stays NON-ML, matching real underwriting governance:
  - Hard-stop eligibility rules must be auditable ("we declined you because
    of X"), not a black box.
  - Premium loads (LAE%, Expense%, profit margin) are company financial
    assumptions set by actuarial/finance leadership, not fitted from data.
  - Composite score weighting (50/20/30) is a business risk-appetite policy
    choice, not something a model should learn on its own.

Coefficients below are decoded 1:1 from the source workbook's Interface
what-if-simulator formulas (see Technical_Notes sheet for the prose
description of the same logic).

This module consumes ML MODEL OUTPUTS (frequency, severity, bind
probability) as inputs - it does not compute them. See scoring/predictors.py
for the model-based frequency/severity/bind predictions.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

TREND = 1.06
DEVELOPMENT = 1.03
LAE_LOAD = 0.10
EXPENSE_LOAD = 0.18
TARGET_PROFIT_MARGIN = 0.05  # configurable 2-7% in the UI
BASE_MARKET_MARGIN = 0.09
PREMIUM_FLOOR = 350
PREMIUM_CEIL = 16000
RENEWAL_BAND_LOW = 0.60
RENEWAL_BAND_HIGH = 1.90
TECHNICAL_FLOOR_PCT = 0.90  # discretionary adjustments can't price >10% under technical

COMPOSITE_THRESHOLDS = {"preferred": 72, "standard": 55, "non_standard": 38}
COMPOSITE_WEIGHTS = {"loss": 0.50, "bind": 0.20, "appetite": 0.30}


def _clip(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


@dataclass
class SubmissionInputs:
    """Raw underwriting inputs for a single submission (matches the
    Interface tab's input fields 1:1)."""
    driver_age: float
    years_licensed: float
    annual_mileage: float
    vehicle_use: str                 # "Business" | "Commute" | "Pleasure"
    territory_type: str              # "Urban" | "Suburban" | "Rural"
    traffic_congestion: float        # 0-10
    vehicle_theft_risk: float        # 0-10
    coverage_lapse_days: float
    undeclared_business_use: bool
    prior_claims_3y: int
    at_fault_claims_3y: int
    prior_claims_5y: int
    moving_violations_3y: int
    speeding_violations_3y: int
    major_violation: bool
    dui_flag: bool
    license_suspension: bool
    telematics_enrolled: bool
    telematics_safety_score: float   # 0-100
    vehicle_value: float
    repairability_score: float       # 0-10
    parts_cost_index: float
    labor_cost_index: float
    medical_cost_index: float
    litigation_environment: float    # 0-10
    luxury_vehicle: bool
    performance_vehicle: bool
    collision_deductible: float
    comprehensive_deductible: float
    cat_exposure_score: float        # 0-10
    state_in_appetite: bool
    prior_total_loss: bool
    confirmed_fraud: bool
    new_renewal: str                 # "New Business" | "Renewal"
    customer_tenure_years: float
    multi_policy: bool
    submission_channel: str          # "Agent" | "Direct" | "Aggregator"
    state: str
    agent_engagement: float          # 0-100
    digital_engagement: float        # 0-100
    quote_revisions: int
    competing_quotes_count: int
    price_sensitivity: str           # "Low" | "Medium" | "High"
    competitor_premium_estimate: Optional[float] = None  # None -> no competitive adjustment applied
    prior_year_premium: Optional[float] = None  # None for new business

    target_profit_margin: float = TARGET_PROFIT_MARGIN  # UI-adjustable


@dataclass
class ScoringResult:
    hard_stop: bool
    hard_stop_reasons: list[str]
    appetite_score: float
    expected_claim_frequency: float
    expected_severity: float
    expected_loss_cost: float
    expected_loss_load: float
    loss_propensity_score: float
    bind_probability: float
    bind_propensity_score: float
    cat_load: float
    risk_load: float
    technical_premium: float
    business_premium_raw: float
    final_quoted_premium: float
    rate_adequacy_flag: bool
    current_rate_premium: float
    expected_loss_ratio: float
    composite_score: float
    uw_risk_band: str
    recommended_action: str


def check_hard_stops(x: SubmissionInputs) -> tuple[bool, list[str]]:
    reasons = []
    if x.dui_flag:
        reasons.append("DUI/DWI conviction")
    if x.license_suspension:
        reasons.append("License suspension")
    if x.coverage_lapse_days > 60:
        reasons.append("Coverage lapse exceeds 60 days")
    if x.confirmed_fraud:
        reasons.append("Confirmed fraud indicator")
    if x.undeclared_business_use:
        reasons.append("Undeclared business use")
    if x.vehicle_value > 100_000:
        reasons.append("Vehicle value exceeds carrier limit ($100,000)")
    if not x.state_in_appetite:
        reasons.append("State outside carrier appetite footprint")
    if x.prior_total_loss:
        reasons.append("Prior total-loss claim")
    return (len(reasons) > 0), reasons


def compute_appetite_score(x: SubmissionInputs, hard_stop: bool) -> float:
    if hard_stop:
        return 0.0
    penalty = (
        4 * x.prior_claims_3y
        + 5 * x.at_fault_claims_3y
        + 3 * x.moving_violations_3y
        + 15 * (1 if x.major_violation else 0)
        + (10 if 30 < x.coverage_lapse_days <= 60 else 0)
        + max(0.0, (x.vehicle_value - 80_000) / 2500)
        + max((x.cat_exposure_score - 7) * 4, 0)
        + (5 if x.driver_age < 25 else 0)
        + (5 if (x.vehicle_use == "Business" and not x.undeclared_business_use) else 0)
        + max((x.litigation_environment - 7) * 3, 0)
        + (6 if x.prior_claims_5y >= 3 else 0)
    )
    return _clip(100 - penalty, 0, 100)


def compute_cat_load(x: SubmissionInputs) -> float:
    return 0.02 + (x.cat_exposure_score - 1) / 9 * 0.06


def compute_risk_load(x: SubmissionInputs) -> float:
    raw = (
        0.02
        + 0.01 * (1 if x.new_renewal == "Renewal" else 0)
        + 0.01 * (1 if x.performance_vehicle else 0)
        + 0.01 * (1 if x.vehicle_value > 60_000 else 0)
        + 0.01 * (1 if x.driver_age < 25 else 0)
    )
    return _clip(raw, 0.02, 0.06)


def compute_technical_premium(expected_loss_load: float, cat_load: float, risk_load: float,
                               target_margin: float) -> float:
    pre_profit_cost = expected_loss_load * (1 + LAE_LOAD + EXPENSE_LOAD + cat_load + risk_load)
    return pre_profit_cost / (1 - target_margin)


def compute_business_premium(x: SubmissionInputs, technical_premium: float,
                              market_adj: float) -> tuple[float, dict]:
    if x.competitor_premium_estimate is not None:
        competitive_adj = _clip(
            ((x.competitor_premium_estimate - technical_premium) / technical_premium) * 0.15,
            -0.05, 0.06,
        )
    else:
        competitive_adj = 0.0
    if x.new_renewal == "Renewal":
        retention_adj = -_clip(
            0.003 * min(x.customer_tenure_years, 10) + 0.01 * (1 if x.multi_policy else 0),
            0, 0.035,
        )
    else:
        retention_adj = 0.0

    channel_adj = {"Aggregator": -0.02, "Agent": 0.01}.get(x.submission_channel, 0.0)

    surcharge_adj = (
        0.03 * (1 if x.major_violation else 0)
        + 0.02 * (1 if x.performance_vehicle else 0)
        + (0.02 if 30 < x.coverage_lapse_days <= 60 else 0)
    )

    bundle_discount = technical_premium * 0.025 if x.multi_policy else 0.0
    safe_driver_discount = (
        technical_premium * 0.02
        if (x.prior_claims_3y == 0 and x.moving_violations_3y == 0
            and not x.major_violation and not x.dui_flag)
        else 0.0
    )
    if x.telematics_enrolled and x.telematics_safety_score >= 70:
        telematics_discount = technical_premium * 0.04
    elif x.telematics_enrolled and x.telematics_safety_score >= 50:
        telematics_discount = technical_premium * 0.015
    else:
        telematics_discount = 0.0

    raw = (
        technical_premium
        * (1 + BASE_MARKET_MARGIN)
        * (1 + market_adj)
        * (1 + competitive_adj)
        * (1 + retention_adj)
        * (1 + channel_adj)
        * (1 + surcharge_adj)
        - (bundle_discount + safe_driver_discount + telematics_discount)
    )
    raw = max(raw, technical_premium * TECHNICAL_FLOOR_PCT)

    breakdown = dict(
        market_adj=market_adj, competitive_adj=competitive_adj, retention_adj=retention_adj,
        channel_adj=channel_adj, surcharge_adj=surcharge_adj, bundle_discount=bundle_discount,
        safe_driver_discount=safe_driver_discount, telematics_discount=telematics_discount,
    )
    return raw, breakdown


def apply_premium_guardrails(business_premium_raw: float, x: SubmissionInputs) -> float:
    if x.new_renewal == "Renewal" and x.prior_year_premium:
        lo = x.prior_year_premium * RENEWAL_BAND_LOW
        hi = x.prior_year_premium * RENEWAL_BAND_HIGH
        guarded = sorted([business_premium_raw, lo, hi])[1]  # MEDIAN of 3
    else:
        guarded = business_premium_raw
    return _clip(guarded, PREMIUM_FLOOR, PREMIUM_CEIL)


def compute_current_rate_premium(x: SubmissionInputs) -> float:
    """Proxy for the coarser 'current rating plan' premium, used only for
    the Expected_Loss_Ratio metric and the Decline-vs-Refer fallback -
    NOT the actual quoted price."""
    if x.new_renewal == "Renewal" and x.prior_year_premium:
        return x.prior_year_premium
    manual_rate = 950 * (
        1
        + 0.10 * (1 if x.driver_age < 25 else 0)
        + 0.05 * (1 if x.driver_age > 70 else 0)
        + 0.12 * x.prior_claims_3y
        + 0.15 * x.at_fault_claims_3y
        + 0.10 * x.moving_violations_3y
        + 0.35 * (1 if x.dui_flag else 0)
        + 0.00002 * x.vehicle_value
        + 0.03 * x.cat_exposure_score
    )
    return _clip(manual_rate, 350, 6000)


def compute_composite_score(loss_propensity_score: float, bind_propensity_score: float,
                             appetite_score: float) -> float:
    score = (
        COMPOSITE_WEIGHTS["loss"] * (100 - loss_propensity_score)
        + COMPOSITE_WEIGHTS["bind"] * bind_propensity_score
        + COMPOSITE_WEIGHTS["appetite"] * appetite_score
    )
    return round(score, 1)


def band_for_score(composite_score: float, hard_stop: bool) -> str:
    if hard_stop:
        return "Hard Stop"
    if composite_score >= COMPOSITE_THRESHOLDS["preferred"]:
        return "Preferred"
    if composite_score >= COMPOSITE_THRESHOLDS["standard"]:
        return "Standard"
    if composite_score >= COMPOSITE_THRESHOLDS["non_standard"]:
        return "Non-Standard"
    return "Decline"


def recommend_action(band: str, hard_stop: bool, final_premium: float, technical_premium: float,
                      current_rate_loss_ratio: float) -> str:
    if hard_stop:
        return "Hard-Stop Route (Decline / Senior UW Review per triggered rule)"
    if band == "Preferred" and final_premium < technical_premium:
        return "Refer to Underwriter (Rate-Adequacy Exception)"
    if band == "Preferred":
        return "Auto-Quote / STP"
    if band == "Standard":
        return "Light UW Review"
    if band == "Non-Standard":
        return "Thorough Verification"
    if current_rate_loss_ratio > 1.05:
        return "Decline"
    return "Refer to Senior Underwriter"


def score_submission(
    x: SubmissionInputs,
    expected_claim_frequency: float,
    expected_severity: float,
    bind_probability: float,
    loss_propensity_score: float,
    market_adj: float = 0.0,
) -> ScoringResult:
    """Combine ML model outputs with the deterministic rules engine to
    produce the full underwriting decision - mirrors the workbook's
    Composite_UW_Risk_Score / Technical_Premium / Business_Premium chain."""

    hard_stop, hard_stop_reasons = check_hard_stops(x)
    appetite_score = compute_appetite_score(x, hard_stop)

    expected_loss_cost = expected_claim_frequency * expected_severity
    expected_loss_load = expected_loss_cost * TREND * DEVELOPMENT

    cat_load = compute_cat_load(x)
    risk_load = compute_risk_load(x)
    technical_premium = compute_technical_premium(
        expected_loss_load, cat_load, risk_load, x.target_profit_margin
    )

    business_premium_raw, _adj_breakdown = compute_business_premium(x, technical_premium, market_adj)
    final_quoted_premium = apply_premium_guardrails(business_premium_raw, x)
    rate_adequacy_flag = final_quoted_premium < technical_premium

    current_rate_premium = compute_current_rate_premium(x)
    expected_loss_ratio = expected_loss_load / current_rate_premium if current_rate_premium else 0.0

    bind_propensity_score = round(bind_probability * 100, 1)

    composite_score = compute_composite_score(loss_propensity_score, bind_propensity_score, appetite_score)
    band = band_for_score(composite_score, hard_stop)
    action = recommend_action(band, hard_stop, final_quoted_premium, technical_premium, expected_loss_ratio)

    return ScoringResult(
        hard_stop=hard_stop,
        hard_stop_reasons=hard_stop_reasons,
        appetite_score=appetite_score,
        expected_claim_frequency=expected_claim_frequency,
        expected_severity=expected_severity,
        expected_loss_cost=expected_loss_cost,
        expected_loss_load=expected_loss_load,
        loss_propensity_score=loss_propensity_score,
        bind_probability=bind_probability,
        bind_propensity_score=bind_propensity_score,
        cat_load=cat_load,
        risk_load=risk_load,
        technical_premium=technical_premium,
        business_premium_raw=business_premium_raw,
        final_quoted_premium=final_quoted_premium,
        rate_adequacy_flag=rate_adequacy_flag,
        current_rate_premium=current_rate_premium,
        expected_loss_ratio=expected_loss_ratio,
        composite_score=composite_score,
        uw_risk_band=band,
        recommended_action=action,
    )
