import numpy as np
from config.sectors import EMISSION_FACTORS, GEI_TARGETS, CBAM_BENCHMARKS
from utils.logger import setup_logger

logger = setup_logger('gei_calculator', 'logs/gei_calculator.log')

class EmissionCalculator:
    def __init__(self, sector, data_quality="high"):
        """
        sector: 'steel', 'cement', 'aluminum', 'fertilizer'
        data_quality: 'high' (all process data available),
                      'medium' (some assumptions),
                      'low' (relying on default benchmarks)
        """
        self.sector = sector.lower()
        self.data_quality = data_quality

        if self.sector not in GEI_TARGETS:
            raise ValueError(f"Unsupported sector: {sector}")

        self.target_gei = GEI_TARGETS[self.sector]
        self.cbam_benchmarks = CBAM_BENCHMARKS[self.sector]

    def calculate_embedded_emissions(self, production_tonnes, fuel_usage=None, electricity_mwh=0.0):
        """
        Calculates direct and indirect emissions.
        Returns a dictionary with base, conservative, and stress estimates.
        """
        if fuel_usage is None:
            fuel_usage = {"coal": 0, "natural_gas": 0}

        # Base Estimate Calculation (Using actual data)
        direct_emissions = sum([qty * EMISSION_FACTORS.get(fuel, 0) for fuel, qty in fuel_usage.items()])
        indirect_emissions = electricity_mwh * EMISSION_FACTORS["grid_electricity"]
        total_base_emissions = direct_emissions + indirect_emissions

        base_gei = total_base_emissions / production_tonnes if production_tonnes > 0 else 0

        # Uncertainty Bands
        if self.data_quality == "high":
            # High confidence, narrow bands
            conservative_gei = base_gei * 0.95
            stress_gei = base_gei * 1.05
        elif self.data_quality == "medium":
            # Moderate confidence
            conservative_gei = base_gei * 0.90
            stress_gei = base_gei * 1.15
        else:
            # Low confidence (e.g., hidden manufacturer data)
            # Fall back heavily towards worst-case CBAM benchmarks
            base_gei = max(base_gei, self.cbam_benchmarks["average"])
            conservative_gei = base_gei * 0.85
            stress_gei = self.cbam_benchmarks["worst_case"]

        return {
            "production_tonnes": production_tonnes,
            "total_tco2e_base": total_base_emissions,
            "gei_conservative": conservative_gei,
            "gei_base": base_gei,
            "gei_stress": stress_gei
        }

    def compute_compliance_gap(self, emissions_data):
        """
        Compares embedded emissions against CCTS and CBAM targets.
        Outputs shortfalls in tCO2e.
        """
        prod = emissions_data["production_tonnes"]
        gei_base = emissions_data["gei_base"]
        gei_stress = emissions_data["gei_stress"]

        # 1. Domestic Compliance Gap (CCTS)
        # Deficit = (Actual GEI - Target GEI) * Production
        # If positive, it's a deficit (shortfall). If negative, it's a surplus.
        ccts_gap_base = (gei_base - self.target_gei) * prod
        ccts_gap_stress = (gei_stress - self.target_gei) * prod

        # 2. Export Compliance Gap (CBAM)
        # For CBAM, we compare against the EU default average benchmark as a proxy
        # for free allocation or threshold.
        cbam_gap_base = (gei_base - self.cbam_benchmarks["average"]) * prod
        cbam_gap_stress = (gei_stress - self.cbam_benchmarks["average"]) * prod

        return {
            "ccts_shortfall_base_tco2e": max(0, ccts_gap_base),
            "ccts_shortfall_stress_tco2e": max(0, ccts_gap_stress),
            "cbam_shortfall_base_tco2e": max(0, cbam_gap_base),
            "cbam_shortfall_stress_tco2e": max(0, cbam_gap_stress),
            "audit_trail": {
                "applied_ccts_target": self.target_gei,
                "applied_cbam_benchmark": self.cbam_benchmarks["average"],
                "data_quality_assumed": self.data_quality
            }
        }
