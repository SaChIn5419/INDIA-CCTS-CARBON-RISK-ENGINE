import sys
import polars as pl
from compliance.gei_calculator import EmissionCalculator
from risk.penalty_engine import ComplianceRiskEngine
from utils.logger import setup_logger, write_progress_report

logger = setup_logger('compliance_integration', 'logs/compliance_integration.log')

def run_compliance_pipeline(sector, data_quality, production, export_ratio, ccts_forecasted_price):
    logger.info(f"Running compliance integration for Sector: {sector.upper()} | Data Quality: {data_quality.upper()}")

    # Initialize components
    gei_calc = EmissionCalculator(sector=sector, data_quality=data_quality)
    risk_engine = ComplianceRiskEngine(ccts_price_forecast_inr=ccts_forecasted_price)

    # 1. Estimate Emissions
    # Mocking fuel usage for demonstration. In production, this would come from the ERP database.
    if sector == "steel":
        fuel_usage = {"coal": production * 0.8} # 0.8 tonnes of coal per tonne of steel
        electricity = production * 0.5 # 0.5 MWh per tonne
    elif sector == "cement":
        fuel_usage = {"coal": production * 0.1}
        electricity = production * 0.1
    else:
        fuel_usage = {}
        electricity = production * 1.5

    emissions = gei_calc.calculate_embedded_emissions(production, fuel_usage, electricity)

    # 2. Compute Gap
    gaps = gei_calc.compute_compliance_gap(emissions)

    # 3. Compute Financial Penalty Exposure & Risk Rank
    exposure = risk_engine.evaluate_financial_exposure(gaps, export_ratio=export_ratio)
    rank_info = risk_engine.determine_risk_rank(exposure, gaps, data_quality=data_quality, export_ratio=export_ratio)

    # Output to logs and markdown
    logger.info(f"Final Risk Rank: {rank_info['risk_rank']}")
    logger.info(f"Total Base Exposure (INR): {exposure['total_exposure_base_inr']:,.2f}")

    report = f"### Compliance & Risk Analysis for {sector.capitalize()} Plant\n\n"
    report += f"**Input Parameters:**\n"
    report += f"- Production: {production:,.0f} tonnes\n"
    report += f"- Export Ratio: {export_ratio:.0%}\n"
    report += f"- Data Quality: {data_quality.capitalize()}\n"
    report += f"- Forecasted CCTS Price (From DL Model): ₹{ccts_forecasted_price:,.2f}\n\n"

    report += f"**Emissions (tCO2e):**\n"
    report += f"- Base Estimate: {emissions['total_tco2e_base']:,.0f}\n"
    report += f"- Conservative GEI: {emissions['gei_conservative']:.2f} | Base GEI: {emissions['gei_base']:.2f} | Stress GEI: {emissions['gei_stress']:.2f}\n\n"

    report += f"**Compliance Gap (Shortfalls):**\n"
    report += f"- Domestic CCTS Gap (Base): {gaps['ccts_shortfall_base_tco2e']:,.0f} tCO2e\n"
    report += f"- Export CBAM Gap (Base): {gaps['cbam_shortfall_base_tco2e']:,.0f} tCO2e\n\n"

    report += f"**Financial Exposure:**\n"
    report += f"- CCTS Penalty Risk (INR): ₹{exposure['domestic_exposure_base_inr']:,.2f} to ₹{exposure['domestic_exposure_stress_inr']:,.2f}\n"
    report += f"- CBAM Export Risk (EUR): €{exposure['export_exposure_base_eur']:,.2f} to €{exposure['export_exposure_stress_eur']:,.2f}\n"
    report += f"- **Total Equivalent Risk (INR)**: ₹{exposure['total_exposure_base_inr']:,.2f}\n\n"

    report += f"**Overall Risk Classification:**\n"
    report += f"- **Rank**: {rank_info['risk_rank']} (Score: {rank_info['risk_score']})\n"
    report += f"- **Audit Trail**: {rank_info['audit_factors']}\n"

    write_progress_report(
        "Phase E: Compliance Engine Integration",
        "Completed",
        report
    )

    return rank_info

if __name__ == "__main__":
    # Test integration with mock forecasted price (e.g., from the Hybrid output)
    forecasted_price_inr = 2500.0

    # Run scenario 1: High risk steel exporter
    run_compliance_pipeline(
        sector="steel",
        data_quality="low",
        production=500_000,
        export_ratio=0.60,
        ccts_forecasted_price=forecasted_price_inr
    )

    # Run scenario 2: Domestic cement producer with good data
    run_compliance_pipeline(
        sector="cement",
        data_quality="high",
        production=1_000_000,
        export_ratio=0.05,
        ccts_forecasted_price=forecasted_price_inr
    )
