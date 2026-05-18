import pytest
from compliance.gei_calculator import EmissionCalculator

def test_emission_calculator_high_quality():
    calc = EmissionCalculator(sector="steel", data_quality="high")

    # 10,000 tonnes of steel
    production = 10000
    # Fuel: 8,000 tonnes of coal (2.4 tCO2e/t) = 19,200 tCO2e
    # Electricity: 5,000 MWh (0.82 tCO2e/MWh) = 4,100 tCO2e
    # Total = 23,300 tCO2e. GEI = 2.33
    fuel = {"coal": 8000}
    elec = 5000

    emissions = calc.calculate_embedded_emissions(production, fuel, elec)

    assert emissions["total_tco2e_base"] == 23300
    assert emissions["gei_base"] == 2.33

    gaps = calc.compute_compliance_gap(emissions)

    # Target GEI for steel is 2.2
    # Deficit = (2.33 - 2.2) * 10000 = 1300 tCO2e
    assert pytest.approx(gaps["ccts_shortfall_base_tco2e"]) == 1300

    # CBAM average benchmark for steel is 2.5
    # Deficit = (2.33 - 2.5) * 10000 = -1700 -> bounded to 0 shortfall
    assert gaps["cbam_shortfall_base_tco2e"] == 0

def test_emission_calculator_low_quality():
    calc = EmissionCalculator(sector="cement", data_quality="low")

    # Low quality should force the stress test to hit the worst_case CBAM benchmark (1.0 for cement)
    emissions = calc.calculate_embedded_emissions(10000, {"coal": 0}, 0)

    assert emissions["gei_stress"] == 1.0

    gaps = calc.compute_compliance_gap(emissions)
    # Target cement = 0.7
    # Stress Deficit = (1.0 - 0.7) * 10000 = 3000
    assert pytest.approx(gaps["ccts_shortfall_stress_tco2e"]) == 3000
