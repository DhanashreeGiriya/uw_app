"""
Maps a raw data row (dict-like: dataframe row or Streamlit form output)
into the rules_engine.SubmissionInputs dataclass. Single source of truth
for the column-name -> field-name mapping, used by both score_book.py
(historical data) and app.py (live form input).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "rules"))
from rules_engine import SubmissionInputs  # noqa: E402


def _yn(v) -> bool:
    """Handles flags stored as bool, numeric 0/1, or 'Y'/'N' strings."""
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(v)
    return str(v).strip().upper() in ("Y", "1", "TRUE")


def row_to_submission_inputs(row, target_profit_margin: float = 0.05,
                              competitor_premium_estimate=None) -> SubmissionInputs:
    return SubmissionInputs(
        driver_age=float(row["Driver_Age"]),
        years_licensed=float(row["Years_Licensed"]),
        annual_mileage=float(row["Annual_Mileage"]),
        vehicle_use=row["Vehicle_Use"],
        territory_type=row["Urban_Suburban_Rural"],
        traffic_congestion=float(row["Traffic_Congestion_Score"]),
        vehicle_theft_risk=float(row["Vehicle_Theft_Risk_Score"]),
        coverage_lapse_days=float(row["Coverage_Lapse_Days"]),
        undeclared_business_use=_yn(row["Undeclared_Business_Use_Flag"]),
        prior_claims_3y=int(row["Prior_Claims_3Y"]),
        at_fault_claims_3y=int(row["At_Fault_Claims_3Y"]),
        prior_claims_5y=int(row["Prior_Claims_5Y"]),
        moving_violations_3y=int(row["Moving_Violations_3Y"]),
        speeding_violations_3y=int(row["Speeding_Violations_3Y"]),
        major_violation=_yn(row["Major_Violation_Flag"]),
        dui_flag=_yn(row["DUI_Flag"]),
        license_suspension=_yn(row["License_Suspension_Flag"]),
        telematics_enrolled=_yn(row["Telematics_Enrolled"]),
        telematics_safety_score=float(row["Telematics_Safety_Score"]) if row["Telematics_Safety_Score"] is not None else 75.0,
        vehicle_value=float(row["Vehicle_Value"]),
        repairability_score=float(row["Repairability_Score"]),
        parts_cost_index=float(row["Parts_Cost_Index"]),
        labor_cost_index=float(row["Labor_Cost_Index"]),
        medical_cost_index=float(row["Medical_Cost_Index"]),
        litigation_environment=float(row["Litigation_Environment_Score"]),
        luxury_vehicle=_yn(row["Luxury_Flag"]),
        performance_vehicle=_yn(row["Performance_Vehicle_Flag"]),
        collision_deductible=float(row["Collision_Deductible"]) if row["Collision_Deductible"] is not None else 500.0,
        comprehensive_deductible=float(row["Comprehensive_Deductible"]) if row["Comprehensive_Deductible"] is not None else 500.0,
        cat_exposure_score=float(row["CAT_Exposure_Score"]),
        state_in_appetite=_yn(row["Carrier_Appetite_State_Flag"]),
        prior_total_loss=_yn(row["Prior_Total_Loss_Flag"]),
        confirmed_fraud=_yn(row["Fraud_Flag"]),
        new_renewal=row["New_Renewal_Label"],
        customer_tenure_years=float(row["Customer_Tenure_Years"]),
        multi_policy=_yn(row["Multi_Policy_Flag"]),
        submission_channel=row["Submission_Channel"],
        state=row["State"],
        agent_engagement=float(row["Agent_Engagement_Score"]) if row["Agent_Engagement_Score"] is not None else 0.0,
        digital_engagement=float(row["Digital_Engagement_Score"]) if row["Digital_Engagement_Score"] is not None else 0.0,
        quote_revisions=int(row["Quote_Revisions"]) if row["Quote_Revisions"] is not None else 0,
        competing_quotes_count=0,
        price_sensitivity=row["Price_Sensitivity_Band"] if row.get("Price_Sensitivity_Band") is not None else "Medium",
        competitor_premium_estimate=competitor_premium_estimate,
        prior_year_premium=float(row["Prior_Year_Premium"]) if row["Prior_Year_Premium"] not in (None,) else None,
        target_profit_margin=target_profit_margin,
    )
