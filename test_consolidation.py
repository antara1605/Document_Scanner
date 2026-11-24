"""
Comprehensive Test Suite for Multi-Document Consolidation & Conflict Handling.
Verifies all 5 required test scenarios:
1. Matching information across documents (Consolidation & Multi-source verification)
2. Conflicting information (Mismatch detection, Non-silent overwriting, Manual review flagging)
3. Missing information (Strict None/null, Zero hallucination)
4. Borrower and Guarantor appearing in different documents (Role resolution & Record segregation)
5. Multiple Guarantors (Independent Guarantor 1 & Guarantor 2 management)
"""

import json
from consolidator import MultiDocumentConsolidator, ReviewStatus, PersonRole

def run_all_consolidation_tests():
    print("=" * 80)
    print("  RUNNING MULTI-DOCUMENT CONSOLIDATION & CONFLICT HANDLING TEST SUITE")
    print("=" * 80)

    consolidator = MultiDocumentConsolidator()

    # ==========================================================================
    # TEST CASE 1: Matching Information Across Documents
    # ==========================================================================
    print("\n" + "=" * 80)
    print("  TEST CASE 1: Matching Information Across Documents")
    print("=" * 80)

    payload_matching = {
        "documents": [
            {
                "document_name": "borrower_pan_card.jpg",
                "document_type": "PAN Card",
                "confidence": 0.96,
                "ocr_text": "INCOME TAX DEPARTMENT\nPermanent Account Number: ABCPM1234F\nName: Rahul Sharma\nDOB: 12/04/1986",
                "fields": {
                    "Full Name": "Rahul Sharma",
                    "PAN Number": "ABCPM1234F",
                    "Date of Birth": "12/04/1986"
                }
            },
            {
                "document_name": "borrower_salary_slip.pdf",
                "document_type": "Salary Slip",
                "confidence": 0.94,
                "ocr_text": "PAYSLIP FOR JULY 2024\nEmployee Name: Rahul Sharma\nPAN: ABCPM1234F\nDOB: 12.04.1986\nGross Salary: 65000",
                "fields": {
                    "Full Name": "Rahul Sharma",
                    "PAN Number": "ABCPM1234F",
                    "Date of Birth": "12.04.1986", # Different format (dots), should normalize & match
                    "Gross Salary": "65000"
                }
            }
        ]
    }

    norm_data_1, audit_trail_1, summary_1 = consolidator.consolidate(payload_matching)

    pan_audit = next((a for a in audit_trail_1 if a.field_path == "borrower.pan_no"), None)
    dob_audit = next((a for a in audit_trail_1 if a.field_path == "borrower.dob"), None)

    print(f"  PAN Value               : {pan_audit.value}")
    print(f"  PAN Review Status       : {pan_audit.review_status}")
    print(f"  PAN Source Documents    : {pan_audit.source_documents}")
    print(f"  PAN Has Conflict        : {pan_audit.has_conflict}")

    print(f"  DOB Value               : {dob_audit.value}")
    print(f"  DOB Review Status       : {dob_audit.review_status}")
    print(f"  DOB Source Documents    : {dob_audit.source_documents}")
    print(f"  DOB Has Conflict        : {dob_audit.has_conflict}")

    assert pan_audit.review_status == ReviewStatus.VERIFIED, "Matching PAN should be VERIFIED_ACROSS_DOCUMENTS"
    assert len(pan_audit.source_documents) == 2, "PAN should cite both source documents"
    assert pan_audit.has_conflict is False
    assert dob_audit.review_status == ReviewStatus.VERIFIED, "DOB in different formats should normalize and match"
    print("  --> PASS: Matching information consolidated with multi-source references without false conflicts.")


    # ==========================================================================
    # TEST CASE 2: Conflicting Information
    # ==========================================================================
    print("\n" + "=" * 80)
    print("  TEST CASE 2: Conflicting Information Across Documents")
    print("=" * 80)

    payload_conflicting = {
        "documents": [
            {
                "document_name": "aadhaar_card.pdf",
                "document_type": "Aadhaar Card",
                "confidence": 0.95,
                "ocr_text": "Government of India\nRahul Sharma\nDOB: 12/04/1986\nMobile: 9832145670",
                "fields": {
                    "Full Name": "Rahul Sharma",
                    "Date of Birth": "12/04/1986",
                    "Mobile No": "9832145670"
                }
            },
            {
                "document_name": "driving_license.jpg",
                "document_type": "Driving License",
                "confidence": 0.92,
                "ocr_text": "UNION OF INDIA DRIVING LICENCE\nHolder: Rahul Sharma\nDOB: 20/12/1985\nMobile: 9434012345",
                "fields": {
                    "Full Name": "Rahul Sharma",
                    "Date of Birth": "20/12/1985", # Discrepancy in Date of Birth
                    "Mobile No": "9434012345"     # Discrepancy in Mobile Number
                }
            }
        ]
    }

    norm_data_2, audit_trail_2, summary_2 = consolidator.consolidate(payload_conflicting)

    dob_conflict = next((a for a in audit_trail_2 if a.field_path == "borrower.dob"), None)
    mob_conflict = next((a for a in audit_trail_2 if a.field_path == "borrower.mobile_no"), None)

    print(f"  DOB Value (Chosen)      : {dob_conflict.value}")
    print(f"  DOB Has Conflict        : {dob_conflict.has_conflict}")
    print(f"  DOB Review Status       : {dob_conflict.review_status}")
    print(f"  DOB Conflicting Options : {dob_conflict.conflicting_values}")

    print(f"  Mobile Has Conflict     : {mob_conflict.has_conflict}")
    print(f"  Mobile Review Status    : {mob_conflict.review_status}")
    print(f"  Mobile Conflicting Opts : {mob_conflict.conflicting_values}")

    assert dob_conflict.has_conflict is True, "Discrepancy in DOB must be flagged as conflict"
    assert dob_conflict.review_status == ReviewStatus.NEEDS_MANUAL_REVIEW
    assert len(dob_conflict.conflicting_values) == 2, "Both conflicting DOB values must be preserved"
    assert mob_conflict.has_conflict is True, "Discrepancy in Mobile must be flagged as conflict"
    print("  --> PASS: Conflicting fields flagged for manual review with full audit details and no silent overwrite.")


    # ==========================================================================
    # TEST CASE 3: Missing Information (Zero Hallucination)
    # ==========================================================================
    print("\n" + "=" * 80)
    print("  TEST CASE 3: Missing Information (Preserving Nulls)")
    print("=" * 80)

    # Payload has only identity fields; loan terms and references are omitted
    payload_sparse = {
        "documents": [
            {
                "document_name": "borrower_pan.pdf",
                "document_type": "PAN Card",
                "confidence": 0.95,
                "fields": {
                    "Full Name": "Rahul Sharma",
                    "PAN Number": "ABCPM1234F"
                }
            }
        ]
    }

    norm_data_3, audit_trail_3, summary_3 = consolidator.consolidate(payload_sparse)

    moratorium_audit = next((a for a in audit_trail_3 if a.field_path == "loan.moratorium_period"), None)
    ref_name_audit = next((a for a in audit_trail_3 if a.field_path == "references[0].name"), None)

    print(f"  Unextracted Moratorium Period : {norm_data_3.loan['moratorium_period'].value}")
    print(f"  Moratorium Audit Status       : {moratorium_audit.review_status}")
    print(f"  Unextracted Reference 1 Name  : {norm_data_3.references[0]['name'].value}")
    print(f"  Reference 1 Audit Status      : {ref_name_audit.review_status}")
    print(f"  Total Missing Fields in Audit : {summary_3['missing_fields']}")

    assert norm_data_3.loan["moratorium_period"].value is None, "Missing fields must remain None"
    assert moratorium_audit.review_status == ReviewStatus.MISSING
    assert norm_data_3.references[0]["name"].value is None
    print("  --> PASS: Missing fields remain strictly None without hallucination.")


    # ==========================================================================
    # TEST CASE 4: Borrower and Guarantor Appearing in Different Documents
    # ==========================================================================
    print("\n" + "=" * 80)
    print("  TEST CASE 4: Borrower and Guarantor in Different Documents")
    print("=" * 80)

    payload_borrower_guarantor = {
        "documents": [
            {
                "document_name": "loan_application_form.pdf",
                "document_type": "Loan Application",
                "confidence": 0.95,
                "ocr_text": "APPLICATION FOR HOUSING LOAN\nApplicant / Borrower: Rahul Sharma\nGuarantor: Amit Sharma\nBranch: Asansol Main",
                "fields": {
                    "Borrower Name": "Rahul Sharma",
                    "Guarantor Name": "Amit Sharma",
                    "Branch": "Asansol Main",
                    "Loan Requested": "1800000"
                }
            },
            {
                "document_name": "borrower_aadhaar.pdf",
                "document_type": "Aadhaar Card",
                "confidence": 0.97,
                "ocr_text": "Government of India\nRahul Sharma\nUID: 1234 5678 9012\nResident Vill: Raniganj",
                "fields": {
                    "Full Name": "Rahul Sharma",
                    "Aadhaar Number": "1234 5678 9012",
                    "Resident Vill": "Raniganj"
                }
            },
            {
                "document_name": "guarantor_pan.jpg",
                "document_type": "PAN Card",
                "confidence": 0.96,
                "ocr_text": "INCOME TAX DEPARTMENT\nName: Amit Sharma\nPAN: XYZPA9876Q\nFather: Ramesh Sharma\nDOB: 05/11/1980",
                "fields": {
                    "Full Name": "Amit Sharma",
                    "PAN Number": "XYZPA9876Q",
                    "Father's Name": "Ramesh Sharma",
                    "Date of Birth": "05/11/1980",
                    "Monthly Income": "48000"
                }
            }
        ]
    }

    norm_data_4, audit_trail_4, summary_4 = consolidator.consolidate(payload_borrower_guarantor)

    borrower_name = norm_data_4.borrower["full_name"].value
    borrower_aadhaar = norm_data_4.borrower["aadhar_no"].value
    guarantor_name = norm_data_4.guarantors[0]["name"].value
    guarantor_father = norm_data_4.guarantors[0]["father_husband_name"].value
    guarantor_income = norm_data_4.guarantors[0]["monthly_income"].value

    print(f"  Borrower Name             : {borrower_name} (from borrower_aadhaar.pdf & loan_application_form.pdf)")
    print(f"  Borrower Aadhaar          : {borrower_aadhaar}")
    print(f"  Guarantor 1 Name          : {guarantor_name} (from guarantor_pan.jpg)")
    print(f"  Guarantor 1 Father        : {guarantor_father}")
    print(f"  Guarantor 1 Monthly Income: {guarantor_income}")

    assert borrower_name == "Rahul Sharma"
    assert guarantor_name == "Amit Sharma"
    assert borrower_name != guarantor_name, "Borrower and Guarantor must remain strictly separate records"
    assert guarantor_father == "Ramesh Sharma", "Guarantor father name must attach to Guarantor 1, not Borrower"
    assert norm_data_4.borrower["father_husband_name"].value is None, "Borrower father name should not be polluted by Guarantor"
    print("  --> PASS: Roles identified contextually; Borrower and Guarantor records strictly segregated.")


    # ==========================================================================
    # TEST CASE 5: Multiple Guarantors (Guarantor 1 and Guarantor 2)
    # ==========================================================================
    print("\n" + "=" * 80)
    print("  TEST CASE 5: Multiple Guarantors (Guarantor 1 and Guarantor 2)")
    print("=" * 80)

    payload_two_guarantors = {
        "documents": [
            {
                "document_name": "guarantor_1_biodata.pdf",
                "document_type": "Guarantee Deed",
                "confidence": 0.95,
                "ocr_text": "LETTER OF GUARANTEE - GUARANTOR 1\nName: Amit Sharma\nOccupation: Service\nMonthly Income: 50000\nResident Vill: Asansol",
                "fields": {
                    "Full Name": "Amit Sharma",
                    "Present Occupation": "Service",
                    "Monthly Income": "50000",
                    "Resident Vill": "Asansol"
                }
            },
            {
                "document_name": "guarantor_2_biodata.pdf",
                "document_type": "Guarantee Deed",
                "confidence": 0.94,
                "ocr_text": "LETTER OF GUARANTEE - GUARANTOR 2\nName: Priya Verma\nOccupation: Business\nMonthly Income: 75000\nResident Vill: Durgapur",
                "fields": {
                    "Full Name": "Priya Verma",
                    "Present Occupation": "Business",
                    "Monthly Income": "75000",
                    "Resident Vill": "Durgapur"
                }
            }
        ]
    }

    norm_data_5, audit_trail_5, summary_5 = consolidator.consolidate(payload_two_guarantors)

    g1 = norm_data_5.guarantors[0]
    g2 = norm_data_5.guarantors[1]

    print(f"  Guarantor 1 Name       : {g1['name'].value} (Source: {g1['name'].source_document})")
    print(f"  Guarantor 1 Occupation : {g1['occupation'].value}")
    print(f"  Guarantor 1 Income     : {g1['monthly_income'].value}")
    print(f"  Guarantor 1 Village    : {g1['resident_address']['vill'].value}")

    print(f"  Guarantor 2 Name       : {g2['name'].value} (Source: {g2['name'].source_document})")
    print(f"  Guarantor 2 Occupation : {g2['occupation'].value}")
    print(f"  Guarantor 2 Income     : {g2['monthly_income'].value}")
    print(f"  Guarantor 2 Village    : {g2['resident_address']['vill'].value}")

    assert g1["name"].value == "Amit Sharma"
    assert g1["monthly_income"].value == "50000"
    assert g2["name"].value == "Priya Verma"
    assert g2["monthly_income"].value == "75000"
    assert g1["name"].value != g2["name"].value, "Guarantor 1 and Guarantor 2 must be distinct"
    print("  --> PASS: Multiple guarantors mapped into their respective isolated containers.")

    print("\n" + "=" * 80)
    print("  ALL 5 CONSOLIDATION & CONFLICT HANDLING TEST SUITES PASSED!")
    print("=" * 80)

if __name__ == "__main__":
    run_all_consolidation_tests()
