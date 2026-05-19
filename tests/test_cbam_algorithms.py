import pytest
from compliance.cbam.algorithms import CBAMCradleToGateEngine, SEFAScalingEngine, CBAMArticle9DeductionEngine

def test_cradle_to_gate_dag():
    engine = CBAMCradleToGateEngine()

    # Simple Precursor: Iron Ore (Emissions = 0.5 tCO2/t)
    engine.add_product("iron_ore", a_eg=0.5, is_complex=False)

    # Precursor: Pig Iron (Emissions = 1.0 tCO2/t)
    engine.add_product("pig_iron", a_eg=1.0, is_complex=True)

    # Final Product: Steel (Emissions = 0.8 tCO2/t)
    engine.add_product("steel", a_eg=0.8, is_complex=True)

    # Dependencies:
    # 1.5t of iron ore -> 1t of pig iron
    engine.add_precursor("pig_iron", "iron_ore", mass_consumed_per_unit=1.5)

    # 1.2t of pig iron -> 1t of steel
    engine.add_precursor("steel", "pig_iron", mass_consumed_per_unit=1.2)

    # SEE of Pig Iron = 1.0 + (1.5 * 0.5) = 1.75
    see_pig_iron = engine.calculate_see("pig_iron")
    assert see_pig_iron == 1.75

    # SEE of Steel = 0.8 + (1.2 * 1.75) = 0.8 + 2.1 = 2.9
    see_steel = engine.calculate_see("steel")
    assert pytest.approx(see_steel) == 2.9

def test_sefa_scaling():
    engine = SEFAScalingEngine()

    # EU ETS benchmark for product is 2.0
    sefa_2026 = engine.calculate_sefa(2026, 2.0)
    # 0.975 * 1.0 * 2.0 = 1.95
    assert pytest.approx(sefa_2026) == 1.95

    sefa_2034 = engine.calculate_sefa(2034, 2.0)
    assert sefa_2034 == 0.0

def test_article_9_deduction():
    engine = CBAMArticle9DeductionEngine()

    # Total emissions = 10,000. Free allowance baseline = 8,000. Price = €10/t. Refunds = €5000.
    eff_price = engine.calculate_effective_price_paid(
        nominal_carbon_price=10.0,
        total_emissions=10000,
        free_allowances_qty=8000,
        refunds_and_comp=5000
    )
    # Chargeable = 2000t. Gross cost = €20,000. Net cost = €15,000.
    # Effective price = 15,000 / 10,000 = 1.5 €/t
    assert pytest.approx(eff_price) == 1.5

    # ITMO Cap test
    capped_itmo = engine.calculate_itmo_cap(10000, 2000)
    # Max allowed is 10% of 10000 = 1000
    assert capped_itmo == 1000
