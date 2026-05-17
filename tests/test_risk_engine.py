import pytest
from risk.penalty_engine import ComplianceRiskEngine
from config.sectors import CBAM_CERTIFICATE_PRICE_EUR, EUR_TO_INR

def test_financial_exposure():
    # Assume DL model forecasted CCC price is 1500 INR
    engine = ComplianceRiskEngine(ccts_price_forecast_inr=1500)

    gaps = {
        "ccts_shortfall_base_tco2e": 1000,
        "ccts_shortfall_stress_tco2e": 1500,
        "cbam_shortfall_base_tco2e": 500,
        "cbam_shortfall_stress_tco2e": 800
    }

    # 50% export ratio
    exposure = engine.evaluate_financial_exposure(gaps, export_ratio=0.5)

    # Domestic = 1000 * 1500 * 2 = 3,000,000 INR
    assert exposure["domestic_exposure_base_inr"] == 3_000_000

    # Export = 500 * 0.5 * 75.36 = 18,840 EUR
    assert pytest.approx(exposure["export_exposure_base_eur"]) == 500 * 0.5 * CBAM_CERTIFICATE_PRICE_EUR

    # Total Base INR = 3M + (18840 * 90) = 3,000,000 + 1,695,600 = 4,695,600
    assert pytest.approx(exposure["total_exposure_base_inr"]) == 3_000_000 + (18840 * EUR_TO_INR)

def test_risk_ranking():
    engine = ComplianceRiskEngine(ccts_price_forecast_inr=1500)

    # Critical risk scenario: massive shortfall, poor data, 100% export
    gaps = {
        "ccts_shortfall_base_tco2e": 20000,
        "ccts_shortfall_stress_tco2e": 25000,
        "cbam_shortfall_base_tco2e": 20000,
        "cbam_shortfall_stress_tco2e": 25000
    }

    exposure = engine.evaluate_financial_exposure(gaps, export_ratio=1.0)
    rank_info = engine.determine_risk_rank(exposure, gaps, data_quality="low", export_ratio=1.0)

    assert rank_info["risk_rank"] == "Critical"
    assert rank_info["audit_factors"]["high_shortfall"] is True
    assert rank_info["audit_factors"]["poor_data"] is True
    assert rank_info["audit_factors"]["high_export"] is True
