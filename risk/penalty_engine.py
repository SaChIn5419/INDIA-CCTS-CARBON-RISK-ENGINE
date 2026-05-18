from config.sectors import CBAM_CERTIFICATE_PRICE_EUR, EUR_TO_INR
from utils.logger import setup_logger

logger = setup_logger('risk_engine', 'logs/risk_engine.log')

class ComplianceRiskEngine:
    def __init__(self, ccts_price_forecast_inr):
        """
        ccts_price_forecast_inr: The forecasted domestic CCC price (e.g., from the Hybrid DL model)
        """
        self.ccts_price_inr = ccts_price_forecast_inr

    def evaluate_financial_exposure(self, gaps, export_ratio=0.0):
        """
        Calculates the penalty cost in INR and EUR.
        export_ratio: float (0.0 to 1.0) indicating % of production exported to EU.
        """
        # 1. Domestic Exposure (CCTS)
        # Indian penalty is often 2x the market price per tonne of deficit (as per blueprint).
        ccts_penalty_multiplier = 2.0

        # We assume the entire domestic shortfall must be covered by domestic CCCs or penalties.
        # Note: If they export, they might not need CCTS for the exported portion,
        # but typically Indian CCTS applies to facility total. We apply to total.
        domestic_exposure_base_inr = gaps["ccts_shortfall_base_tco2e"] * self.ccts_price_inr * ccts_penalty_multiplier
        domestic_exposure_stress_inr = gaps["ccts_shortfall_stress_tco2e"] * self.ccts_price_inr * ccts_penalty_multiplier

        # 2. Export Exposure (CBAM)
        # Only applies to the fraction of production exported.
        # 1 CBAM certificate = 1 tCO2e. Price is in EUR.
        cbam_shortfall_base_exported = gaps["cbam_shortfall_base_tco2e"] * export_ratio
        cbam_shortfall_stress_exported = gaps["cbam_shortfall_stress_tco2e"] * export_ratio

        export_exposure_base_eur = cbam_shortfall_base_exported * CBAM_CERTIFICATE_PRICE_EUR
        export_exposure_stress_eur = cbam_shortfall_stress_exported * CBAM_CERTIFICATE_PRICE_EUR

        # Convert EUR to INR for total impact
        export_exposure_base_inr = export_exposure_base_eur * EUR_TO_INR
        export_exposure_stress_inr = export_exposure_stress_eur * EUR_TO_INR

        total_exposure_base_inr = domestic_exposure_base_inr + export_exposure_base_inr
        total_exposure_stress_inr = domestic_exposure_stress_inr + export_exposure_stress_inr

        return {
            "domestic_exposure_base_inr": domestic_exposure_base_inr,
            "domestic_exposure_stress_inr": domestic_exposure_stress_inr,
            "export_exposure_base_eur": export_exposure_base_eur,
            "export_exposure_stress_eur": export_exposure_stress_eur,
            "total_exposure_base_inr": total_exposure_base_inr,
            "total_exposure_stress_inr": total_exposure_stress_inr
        }

    def determine_risk_rank(self, exposure_data, gaps, data_quality, export_ratio):
        """
        Ranks the company from Low to Critical risk.
        """
        score = 0

        # 1. Emissions Shortfall Size
        if gaps["ccts_shortfall_base_tco2e"] > 10000:
            score += 3
        elif gaps["ccts_shortfall_base_tco2e"] > 1000:
            score += 1

        # 2. Data Quality (Poor data implies hidden risks)
        if data_quality == "low":
            score += 3
        elif data_quality == "medium":
            score += 1

        # 3. Export Exposure
        if export_ratio > 0.5:
            score += 3
        elif export_ratio > 0.1:
            score += 1

        # 4. Financial Impact Thresholds (Arbitrary for ranking: e.g., > 1 Cr INR is bad)
        if exposure_data["total_exposure_base_inr"] > 10_000_000: # 1 Crore INR
            score += 2

        if score >= 8:
            rank = "Critical"
        elif score >= 5:
            rank = "High"
        elif score >= 2:
            rank = "Moderate"
        else:
            rank = "Low"

        return {
            "risk_rank": rank,
            "risk_score": score,
            "audit_factors": {
                "high_shortfall": gaps["ccts_shortfall_base_tco2e"] > 10000,
                "poor_data": data_quality == "low",
                "high_export": export_ratio > 0.5,
                "high_financial_impact": exposure_data["total_exposure_base_inr"] > 10_000_000
            }
        }
