import pytest
from lxml import etree
from compliance.cbam.xml_serializer import CBAMXMLSerializer

def test_xml_serialization_and_validation():
    serializer = CBAMXMLSerializer()

    goods_data = [
        {
            "cn_code": "7208 39 00", # Specific steel product CN Code
            "origin_country": "IN",
            "quantity": 5000.0,
            "see": 2.5000,
            "sefa": 1.9500,
            "effective_price": 5.0,
            "certificates_to_surrender": 2750.0
        }
    ]

    root, xml_str = serializer.generate_xml_payload(
        declarant_id="EU_CBAM_12345",
        reporting_year=2026,
        goods_data=goods_data
    )

    assert b"CBAMDeclaration" in xml_str
    assert b"7208 39 00" in xml_str

    is_valid, errors = serializer.validate_payload(root)
    assert is_valid is True
    assert errors is None

def test_xml_validation_failure():
    serializer = CBAMXMLSerializer()

    # Create invalid XML manually (missing required elements)
    invalid_xml_str = """<?xml version="1.0" encoding="UTF-8"?>
    <CBAMDeclaration>
      <DeclarantID>BAD_123</DeclarantID>
      <!-- Missing ReportingYear and Goods -->
    </CBAMDeclaration>
    """
    root = etree.XML(invalid_xml_str.encode('utf-8'))

    is_valid, errors = serializer.validate_payload(root)
    assert is_valid is False
    assert errors is not None
