"""
Static targets and parameters for CCTS and CBAM compliance.
"""

# Domestic CCTS GEI (Greenhouse Gas Emission Intensity) Targets
# These are placeholder targets representing typical tCO2e per tonne of output
# based on India's PAT/CCTS framework for hard-to-abate sectors.
GEI_TARGETS = {
    "steel": 2.2,     # 2.2 tCO2e per tonne of steel
    "cement": 0.7,    # 0.7 tCO2e per tonne of cement
    "aluminum": 12.5, # 12.5 tCO2e per tonne of aluminum
    "fertilizer": 1.5 # 1.5 tCO2e per tonne of fertilizer
}

# Emission Factors (tCO2e per unit)
EMISSION_FACTORS = {
    "coal": 2.4,          # tCO2e per tonne of coal
    "natural_gas": 0.002, # tCO2e per cubic meter
    "grid_electricity": 0.82 # tCO2e per MWh (Average Indian grid factor)
}

# CBAM Default Benchmarks (EU ETS reference values for complex goods)
# If manufacturer data is missing, we use these worst-case or average benchmarks.
CBAM_BENCHMARKS = {
    "steel": {"average": 2.5, "worst_case": 3.0},
    "cement": {"average": 0.8, "worst_case": 1.0},
    "aluminum": {"average": 14.0, "worst_case": 18.0},
    "fertilizer": {"average": 1.8, "worst_case": 2.5}
}

# Exchange Rates & Certificate Prices
EUR_TO_INR = 90.0
CBAM_CERTIFICATE_PRICE_EUR = 75.36 # First 2026 CBAM certificate price
