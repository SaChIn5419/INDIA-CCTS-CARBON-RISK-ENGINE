import numpy as np
from compliance.cbam.algorithms import CBAMCradleToGateEngine, SEFAScalingEngine, CBAMArticle9DeductionEngine
from utils.logger import setup_logger

logger = setup_logger('cbam_penalties', 'logs/cbam_penalties.log')

# CBAM Default Definitions
CBAM_DEFINITIVE_PENALTY_EUR = 100.0  # €100/tCO2e fine for underreporting
CBAM_CERTIFICATE_PRICE_EUR = 75.36   # E.g. First 2026 auction price

class CBAMDefinitiveRiskEngine:
    def __init__(self):
        self.sefa_engine = SEFAScalingEngine()
        self.deduction_engine = CBAMArticle9DeductionEngine()

    def get_default_markup(self, year, sector="steel"):
        """
        Calculates punitive default value markups for missing data.
        Markups escalate from 10% in 2026 to 30% by 2028 for key sectors.
        """
        # Linear ramp from 2026 (10%) to 2028 (30%)
        if year <= 2026:
            return 1.10
        elif year == 2027:
            return 1.20
        else:
            return 1.30

    def get_geopolitical_benchmark(self, origin="unknown", sector="steel"):
        """
        Geopolitical Arbitrage: If origin is unknown, use the worst-case benchmark
        globally (e.g., China's rate for certain steel tubes over India's).
        """
        benchmarks = {
            "steel": {
                "india": 4.173,
                "china": 9.133,
                "unknown": 9.133 # Defaults to worst offender
            },
            "cement": {
                "india": 0.8,
                "china": 1.1,
                "unknown": 1.1
            }
        }

        sec_bms = benchmarks.get(sector.lower(), {"unknown": 10.0})
        return sec_bms.get(origin.lower(), sec_bms["unknown"])

    def validate_materiality(self, reported_emissions, verified_emissions):
        """
        Strict Audit Materiality Pre-screening.
        Variances > 5% trigger audit failures and severe fines.
        """
        variance = abs(reported_emissions - verified_emissions) / (verified_emissions + 1e-9)
        passed = variance <= 0.05
        return passed, variance

    def validate_materiality_sefa(self, reported_sefa, verified_sefa):
        """
        Applies the 5% materiality threshold to SEFA as requested.
        """
        variance = abs(reported_sefa - verified_sefa) / (verified_sefa + 1e-9)
        passed = variance <= 0.05
        return passed, variance

    def calculate_cbam_liability(self,
                                 product_total_see,
                                 production_exported,
                                 year,
                                 eu_benchmark,
                                 origin,
                                 data_missing=False,
                                 effective_price_paid_eur=0.0):
        """
        Calculates the definitive CBAM certificate requirement and net financial liability.
        """
        # 1. Apply punitive markups or worst-case defaults if data is missing
        if data_missing:
            worst_case_bm = self.get_geopolitical_benchmark(origin="unknown") # Punitive origin
            markup = self.get_default_markup(year)
            final_see = worst_case_bm * markup
            logger.warning(f"Data missing! Applied worst-case benchmark ({worst_case_bm}) with {markup}x markup. Final SEE={final_see}")
        else:
            final_see = product_total_see

        # 2. Calculate SEFA reduction
        sefa = self.sefa_engine.calculate_sefa(year, eu_benchmark)

        # 3. Calculate gross CBAM obligation (tonnes of CO2e per unit of product)
        # Obligation = SEE - SEFA (must be >= 0)
        unit_obligation = max(0.0, final_see - sefa)

        # Total certificates required
        total_certificates_required = unit_obligation * production_exported

        # 4. Article 9 deduction (Monetary)
        # Value of certificates before domestic deduction
        gross_liability_eur = total_certificates_required * CBAM_CERTIFICATE_PRICE_EUR

        # Value already paid domestically on the EXPORTED portion
        domestic_deduction_eur = production_exported * final_see * effective_price_paid_eur

        net_liability_eur = max(0.0, gross_liability_eur - domestic_deduction_eur)

        # Output Certificates to buy = Net Liability / CBAM Price
        final_certificates_to_surrender = net_liability_eur / CBAM_CERTIFICATE_PRICE_EUR

        return {
            "final_see": final_see,
            "sefa_applied": sefa,
            "unit_obligation": unit_obligation,
            "gross_certificates_required": total_certificates_required,
            "net_certificates_to_surrender": final_certificates_to_surrender,
            "net_liability_eur": net_liability_eur
        }

    def evaluate_audit_risk(self, reported_see, true_cradle_to_gate_see, reported_sefa, true_sefa, production_exported):
        """
        Models the worst-case CVaR penalty scenario utilizing the definitive €100/tCO2e fine
        for gate-to-gate vs cradle-to-gate underreporting boundary errors and SEFA overreporting.
        """
        passed_see, variance_see = self.validate_materiality(reported_see, true_cradle_to_gate_see)
        passed_sefa, variance_sefa = self.validate_materiality_sefa(reported_sefa, true_sefa)

        fine_eur = 0.0
        audit_passed = passed_see and passed_sefa

        if not audit_passed:
            # Underreported emissions trigger the €100/t fine
            underreported_emissions = max(0.0, true_cradle_to_gate_see - reported_see) * production_exported

            # Overreported SEFA triggers a fine because it illegally reduces the certificate obligation
            overreported_sefa = max(0.0, reported_sefa - true_sefa) * production_exported

            total_violating_tonnes = underreported_emissions + overreported_sefa

            fine_eur = total_violating_tonnes * CBAM_DEFINITIVE_PENALTY_EUR
            logger.error(f"Materiality Audit Failed! SEE Variance: {variance_see:.2%}. SEFA Variance: {variance_sefa:.2%}. Penalty: €{fine_eur:,.2f}")

        return {
            "audit_passed": audit_passed,
            "variance_see": variance_see,
            "variance_sefa": variance_sefa,
            "punitive_fines_eur": fine_eur
        }
