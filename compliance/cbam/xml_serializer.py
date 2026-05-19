import os
from lxml import etree
from utils.logger import setup_logger

logger = setup_logger('cbam_xml', 'logs/cbam_xml.log')

class CBAMXMLSerializer:
    def __init__(self):
        # A simple XSD Schema approximation for CBAM compliance payload validation
        self.xsd_schema_str = """<?xml version="1.0" encoding="UTF-8"?>
        <xs:schema xmlns:xs="http://www.w3.org/2001/XMLSchema" elementFormDefault="qualified">
          <xs:element name="CBAMDeclaration">
            <xs:complexType>
              <xs:sequence>
                <xs:element name="DeclarantID" type="xs:string"/>
                <xs:element name="ReportingYear" type="xs:integer"/>
                <xs:element name="Goods">
                  <xs:complexType>
                    <xs:sequence>
                      <xs:element name="Good" maxOccurs="unbounded">
                        <xs:complexType>
                          <xs:sequence>
                            <xs:element name="CNCode" type="xs:string"/>
                            <xs:element name="OriginCountry" type="xs:string"/>
                            <xs:element name="QuantityImported" type="xs:decimal"/>
                            <xs:element name="SpecificEmbeddedEmissions" type="xs:decimal"/>
                            <xs:element name="N2O_Emissions" type="xs:decimal" minOccurs="0"/>
                            <xs:element name="PFC_Emissions" type="xs:decimal" minOccurs="0"/>
                            <xs:element name="SEFA" type="xs:decimal"/>
                            <xs:element name="EffectiveCarbonPricePaid" type="xs:decimal"/>
                            <xs:element name="CertificatesToSurrender" type="xs:decimal"/>
                          </xs:sequence>
                        </xs:complexType>
                      </xs:element>
                    </xs:sequence>
                  </xs:complexType>
                </xs:element>
              </xs:sequence>
            </xs:complexType>
          </xs:element>
        </xs:schema>
        """
        schema_root = etree.XML(self.xsd_schema_str.encode('utf-8'))
        self.xml_schema = etree.XMLSchema(schema_root)

    def generate_xml_payload(self, declarant_id, reporting_year, goods_data):
        """
        Serializes a Python dictionary of goods data into the EU CBAM XML format.
        """
        root = etree.Element("CBAMDeclaration")

        etree.SubElement(root, "DeclarantID").text = str(declarant_id)
        etree.SubElement(root, "ReportingYear").text = str(reporting_year)

        goods_el = etree.SubElement(root, "Goods")

        for good in goods_data:
            good_node = etree.SubElement(goods_el, "Good")
            etree.SubElement(good_node, "CNCode").text = str(good.get("cn_code", "UNKNOWN"))
            etree.SubElement(good_node, "OriginCountry").text = str(good.get("origin_country", "UNKNOWN"))
            etree.SubElement(good_node, "QuantityImported").text = str(good.get("quantity", 0.0))
            etree.SubElement(good_node, "SpecificEmbeddedEmissions").text = f"{good.get('see', 0.0):.6f}"

            # Multi-Gas support: only add if explicitly passed to avoid empty tag schema issues unless desired
            if 'n2o_emissions' in good:
                etree.SubElement(good_node, "N2O_Emissions").text = f"{good.get('n2o_emissions', 0.0):.6f}"
            if 'pfc_emissions' in good:
                etree.SubElement(good_node, "PFC_Emissions").text = f"{good.get('pfc_emissions', 0.0):.6f}"

            etree.SubElement(good_node, "SEFA").text = f"{good.get('sefa', 0.0):.6f}"
            etree.SubElement(good_node, "EffectiveCarbonPricePaid").text = f"{good.get('effective_price', 0.0):.2f}"
            etree.SubElement(good_node, "CertificatesToSurrender").text = f"{good.get('certificates_to_surrender', 0.0):.6f}"

        xml_string = etree.tostring(root, pretty_print=True, xml_declaration=True, encoding="UTF-8")
        return root, xml_string

    def validate_payload(self, xml_root):
        """
        Validates the generated XML payload against the XSD Schema.
        Returns (is_valid, error_log)
        """
        is_valid = self.xml_schema.validate(xml_root)
        error_log = self.xml_schema.error_log if not is_valid else None

        if is_valid:
            logger.info("XML payload successfully validated against CBAM XSD.")
        else:
            logger.error(f"XML validation failed: {error_log}")

        return is_valid, error_log

    def export_to_file(self, xml_string, filename="reports/cbam_declaration.xml"):
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, "wb") as f:
            f.write(xml_string)
        logger.info(f"Exported CBAM XML to {filename}")
        return filename
