import pytest
from compliance.cbam.risk_engine import CBAMDefinitiveRiskEngine, CBAM_DEFINITIVE_PENALTY_EUR

def test_markup_and_geopolitics():
    engine = CBAMDefinitiveRiskEngine()

    # Check default markup scaling
    assert engine.get_default_markup(2026) == 1.10
    assert engine.get_default_markup(2028) == 1.30

    # Check geopolitical defaults
    assert engine.get_geopolitical_benchmark("india", "steel") == 4.173
    assert engine.get_geopolitical_benchmark("unknown", "steel") == 9.133

def test_cbam_liability_calculation():
    engine = CBAMDefinitiveRiskEngine()

    # Scenario: Clean data, no missing data.
    # SEE = 3.0, SEFA (2026) = ~1.95 (0.975 * benchmark 2.0).
    # Export = 1000 tonnes.
    # Effective Domestic Price Paid = €5.0
    res = engine.calculate_cbam_liability(
        product_total_see=3.0,
        production_exported=1000,
        year=2026,
        eu_benchmark=2.0,
        origin="india",
        data_missing=False,
        effective_price_paid_eur=5.0
    )

    assert res["sefa_applied"] == 1.95
    # Obligation per unit = 3.0 - 1.95 = 1.05
    assert pytest.approx(res["unit_obligation"]) == 1.05
    # Gross certificates = 1.05 * 1000 = 1050
    assert pytest.approx(res["gross_certificates_required"]) == 1050

    # Gross liability = 1050 * 75.36 = 79,128 EUR
    # Deduction = 1000 * 3.0 * 5.0 = 15,000 EUR
    # Net Liability = 79,128 - 15,000 = 64,128 EUR
    assert pytest.approx(res["net_liability_eur"]) == 64128.0

def test_audit_risk_and_materiality():
    engine = CBAMDefinitiveRiskEngine()

    # User reported gate-to-gate (1.5), but true cradle-to-gate was (3.0)
    # User accurately reported SEFA (2.0) vs true SEFA (2.0)
    # Variance_see = 50% (fails 5% materiality threshold)
    audit = engine.evaluate_audit_risk(
        reported_see=1.5,
        true_cradle_to_gate_see=3.0,
        reported_sefa=2.0,
        true_sefa=2.0,
        production_exported=1000
    )

    assert audit["audit_passed"] is False
    assert pytest.approx(audit["variance_see"], 0.01) == 0.50
    assert audit["variance_sefa"] == 0.0
    # Underreported = 1.5 * 1000 = 1500t
    # Fine = 1500 * €100 = €150,000
    assert audit["punitive_fines_eur"] == 150000.0

    # Pass scenario
    audit_pass = engine.evaluate_audit_risk(2.95, 3.0, 1.95, 2.0, 1000)
    assert audit_pass["audit_passed"] is True
    assert audit_pass["punitive_fines_eur"] == 0.0
