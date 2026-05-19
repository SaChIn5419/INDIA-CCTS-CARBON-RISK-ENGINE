import sys
from compliance.cbam.algorithms import CBAMCradleToGateEngine, CBAMArticle9DeductionEngine
from compliance.cbam.risk_engine import CBAMDefinitiveRiskEngine
from compliance.cbam.xml_serializer import CBAMXMLSerializer
from utils.logger import setup_logger, write_progress_report

logger = setup_logger('cbam_integration', 'logs/cbam_integration.log')

def run_cbam_definitive_pipeline():
    logger.info("Running EU CBAM Definitive Regime Pipeline (Phase F)...")

    # 1. Supply Chain DAG calculation (Cradle-to-gate)
    c2g_engine = CBAMCradleToGateEngine()

    # Define complex supply chain
    # Indian Steel exporter buys Pig Iron and turns it into final Steel Coils.
    c2g_engine.add_product("iron_ore", a_eg=0.5, is_complex=False)
    c2g_engine.add_product("pig_iron", a_eg=1.0, is_complex=True)
    c2g_engine.add_product("steel_coil", a_eg=0.8, is_complex=True)

    c2g_engine.add_precursor("pig_iron", "iron_ore", mass_consumed_per_unit=1.5) # 1.5t ore -> 1t pig iron
    c2g_engine.add_precursor("steel_coil", "pig_iron", mass_consumed_per_unit=1.2) # 1.2t pig iron -> 1t steel

    final_see = c2g_engine.calculate_see("steel_coil")
    logger.info(f"Cradle-to-Gate Specific Embedded Emissions for steel_coil: {final_see:.4f} tCO2/t")

    # 2. Article 9 Deductions (CCTS Price Paid)
    art9_engine = CBAMArticle9DeductionEngine()
    # E.g. Indian CCTS carbon price is 25 EUR. Exporter's facility had 10,000t total emissions,
    # but 8,000 were free under the baseline. They also got 5,000 EUR in indirect compensations.
    effective_price = art9_engine.calculate_effective_price_paid(
        nominal_carbon_price=25.0,
        total_emissions=10000,
        free_allowances_qty=8000,
        refunds_and_comp=5000
    )
    logger.info(f"Effective Article 9 Deduction Price: {effective_price:.2f} EUR/t")

    # 3. Liability & Risk Calculation
    risk_engine = CBAMDefinitiveRiskEngine()

    # Assume 10,000 tonnes of steel exported to EU in 2026.
    export_qty = 10000.0

    liability_data = risk_engine.calculate_cbam_liability(
        product_total_see=final_see,
        production_exported=export_qty,
        year=2026,
        eu_benchmark=2.0,
        origin="india",
        data_missing=False,
        effective_price_paid_eur=effective_price
    )

    # 4. Materiality Audit Pre-Screening
    # Suppose they incorrectly reported just the gate-to-gate portion (0.8) instead of 2.9
    # Suppose they also over-reported SEFA (2.0 instead of 1.95) to cheat
    audit_data = risk_engine.evaluate_audit_risk(
        reported_see=0.8,
        true_cradle_to_gate_see=final_see,
        reported_sefa=2.0,
        true_sefa=liability_data["sefa_applied"],
        production_exported=export_qty
    )

    # 5. Automated XML Serialization
    xml_serializer = CBAMXMLSerializer()

    goods_declaration = [{
        "cn_code": "7208 39 00",
        "origin_country": "IN",
        "quantity": export_qty,
        "see": liability_data["final_see"],
        "sefa": liability_data["sefa_applied"],
        "effective_price": effective_price,
        "certificates_to_surrender": liability_data["net_certificates_to_surrender"]
    }]

    xml_root, xml_string = xml_serializer.generate_xml_payload(
        declarant_id="IND_STEEL_999",
        reporting_year=2026,
        goods_data=goods_declaration
    )

    is_valid, errors = xml_serializer.validate_payload(xml_root)
    if is_valid:
        xml_serializer.export_to_file(xml_string)

    # Write Report
    report = "### Phase F: EU CBAM Definitive Regime Architecture\n\n"
    report += "**1. Cradle-to-Gate DAG Calculation**\n"
    report += f"- Product: Steel Coil\n"
    report += f"- Specific Embedded Emissions (SEE): {final_see:.4f} tCO2/t *(Includes Iron Ore -> Pig Iron -> Steel)*\n\n"

    report += "**2. Financial Liability & SEFA (2026)**\n"
    report += f"- Total Exported: {export_qty:,.0f} tonnes\n"
    report += f"- Applied SEFA (97.5% of 2.0 Benchmark): {liability_data['sefa_applied']:.4f}\n"
    report += f"- Effective Domestic Price Paid (Art. 9): €{effective_price:.2f}/t\n"
    report += f"- **Net Certificates to Surrender**: {liability_data['net_certificates_to_surrender']:,.2f}\n"
    report += f"- **Net Financial Liability**: €{liability_data['net_liability_eur']:,.2f}\n\n"

    report += "**3. Audit Materiality Pre-Screening**\n"
    report += f"- Reported SEE (Gate-to-Gate error): 0.8\n"
    report += f"- True SEE (Cradle-to-Gate): {final_see:.4f}\n"
    report += f"- SEE Variance: {audit_data['variance_see']:.2%}\n"
    report += f"- SEFA Variance: {audit_data['variance_sefa']:.2%}\n"
    report += f"- Audit Passed: {audit_data['audit_passed']}\n"
    report += f"- **Worst-Case Definitive Penalty Exposure**: €{audit_data['punitive_fines_eur']:,.2f}\n\n"

    report += "**4. XSD Payload Validation**\n"
    report += f"- XML Generated: Yes\n"
    report += f"- Passed EU CBAM Schema Validation: {is_valid}\n"
    report += "- Output File: `reports/cbam_declaration.xml`\n"

    write_progress_report(
        "Phase F: EU CBAM Definitive Compliance Integration",
        "Completed",
        report
    )

if __name__ == "__main__":
    run_cbam_definitive_pipeline()
