"""
End-to-End Verification Test for Excel Field Mapping Engine.
Tests populate_excel() with sample loan application data against the real
WBGB HBL LOAN (2).xlsm template.
"""

import os
import openpyxl
from normalizer import NormalizationEngine
from excel_service import populate_excel

def run_excel_mapping_tests():
    print("=" * 80)
    print("  RUNNING EXCEL FIELD MAPPING ENGINE VERIFICATION")
    print("=" * 80)

    template_path = "D:/dscan/template_copy/WBGB_HBL_LOAN.xlsm"
    output_dir = "D:/dscan/output"
    output_path = os.path.join(output_dir, "WBGB_HBL_LOAN_Subhasish_Mukherjee.xlsm")

    assert os.path.exists(template_path), f"Template must exist at {template_path}"

    # Record template modification time before test
    template_mtime_before = os.path.getmtime(template_path)

    # 1. Prepare Realistic Multi-Document Extraction Payload
    sample_extraction_payload = {
        "success": True,
        "processed_files_count": 4,
        "processed_files": [
            "borrower_aadhaar.pdf",
            "borrower_salary_slip.pdf",
            "guarantor1_pan_card.jpg",
            "property_sale_deed.pdf"
        ],
        "engine_used": "Groq AI (llama-3.3-70b-versatile)",
        "documents": [
            {
                "document_name": "borrower_aadhaar.pdf",
                "document_type": "Aadhaar Card",
                "confidence": 0.98,
                "ocr_text": "Government of India\nSubhasish Mukherjee\nDOB: 15/08/1988\nGender: MALE\nAddress: Vill - Kalipur, PO - Kalipur, Dist - Burdwan, PIN - 713101\nUID: 5412 8745 9632",
                "fields": {
                    "Full Name": "Subhasish Mukherjee",
                    "Father's Name": "Ramkrishna Mukherjee",
                    "Date of Birth": "15/08/1988",
                    "Gender": "MALE",
                    "Aadhaar Number": "5412 8745 9632",
                    "Mobile No": "9832145670",
                    "Resident Vill": "Kalipur",
                    "Resident Post": "Kalipur",
                    "Resident Dist": "Burdwan",
                    "Resident Pin": "713101",
                    "Educational Qualification": "Post Graduate"
                }
            },
            {
                "document_name": "borrower_salary_slip.pdf",
                "document_type": "Salary Slip",
                "confidence": 0.95,
                "ocr_text": "PASCHIM BANGA GRAMIN VIKASH\nEmployee: Subhasish Mukherjee\nPAN: ABCPM1234F\nMonth: July 2024\nBasic: 45000\nDA: 12500\nHRA: 4500\nGross Pay: 62000\nPF: 5400\nIncome Tax: 3200\nSalary A/C No: 50100458921\nBranch: Burdwan Main",
                "fields": {
                    "Full Name": "Subhasish Mukherjee",
                    "PAN Number": "ABCPM1234F",
                    "Salary Month": "July 2024",
                    "Gross Salary": "62000",
                    "Basic Pay": "45000",
                    "D.A": "12500",
                    "HRA": "4500",
                    "P.F": "5400",
                    "Income Tax": "3200",
                    "Salary AC No": "50100458921",
                    "Branch": "Burdwan Main",
                    "Application Date": "01.07.2024",
                    "Loan Requested": "2500000",
                    "Tenure Months": "240",
                    "Rate of Interest": "8.5",
                    "Applicant Designation": "Assistant Teacher",
                    "Name of the Organization": "Govt High School",
                    "Length of Service": "10 Years"
                }
            },
            {
                "document_name": "guarantor1_pan_card.jpg",
                "document_type": "PAN Card",
                "confidence": 0.94,
                "ocr_text": "INCOME TAX DEPARTMENT\nName: Animesh Banerjee\nFather's Name: Ramkrishna Banerjee\nDOB: 10/02/1982\nPermanent Account Number: BXYZB5678K",
                "fields": {
                    "Full Name": "Animesh Banerjee",
                    "Father's Name": "Ramkrishna Banerjee",
                    "Date of Birth": "10/02/1982",
                    "Monthly Income": "55000",
                    "Mobile No": "9434012345",
                    "Present Occupation": "Government Employee",
                    "Work Vill": "Burdwan Town",
                    "Work Dist": "Burdwan"
                }
            },
            {
                "document_name": "property_sale_deed.pdf",
                "document_type": "Sale Deed",
                "confidence": 0.93,
                "ocr_text": "DEED OF SALE\nDeed No: 4521/2021\nSub Registry Office: Burdwan\nMouza: Mirzapur, JL No: 45\nPlot / Dag No: 512, Khatian No: 1204\nArea: 4 Decimal\nPurchase Price: 1800000\nBounded on North: 12ft Wide Road\nBounded on South: Plot No 513\nBounded on East: Land of Ghosh\nBounded on West: Land of Paul",
                "fields": {
                    "Deed Number": "4521/2021",
                    "Sub Registry Office": "Burdwan",
                    "Mouza": "Mirzapur",
                    "JL No": "45",
                    "Plot / Dag No": "512",
                    "Khatian No": "1204",
                    "Area": "4 Decimal",
                    "Purchase Price": "1800000",
                    "North": "12ft Wide Road",
                    "South": "Plot No 513",
                    "East": "Land of Ghosh",
                    "West": "Land of Paul",
                    "Construction Cost": "2200000"
                }
            }
        ]
    }

    # 2. Run Normalization
    engine = NormalizationEngine()
    normalized_data = engine.normalize(sample_extraction_payload)

    # 3. Execute populate_excel()
    report = populate_excel(
        normalized_data=normalized_data,
        template_path=template_path,
        output_path=output_path,
        overwrite_existing_output=True
    )

    # 4. Verify Mapping Report
    print("\n--- MAPPING REPORT SUMMARY ---")
    print(f"  Success                  : {report['success']}")
    print(f"  Total Fields Configured  : {report['total_fields_configured']}")
    print(f"  Total Fields Populated   : {report['total_fields_mapped']}")
    print(f"  Missing Mandatory Fields : {len(report['missing_mandatory_fields'])}")
    print(f"  Missing Optional Fields  : {len(report['missing_optional_fields'])}")
    print(f"  Skipped (Protected Cells): {len(report['skipped_fields'])}")
    print(f"  Uncertain Fields         : {len(report['uncertain_fields'])}")
    print(f"  Validation Warnings      : {len(report['validation_warnings'])}")
    print(f"  Downstream Sheets Safe   : {len(report['downstream_sheets_preserved'])}")

    assert report["success"] is True, "populate_excel() should report success=True"
    assert report["total_fields_mapped"] >= 30, f"Expected at least 30 fields mapped, got {report['total_fields_mapped']}"

    # 5. Verify Original Template was NOT Modified
    template_mtime_after = os.path.getmtime(template_path)
    assert template_mtime_before == template_mtime_after, "CRITICAL ERROR: Original template was modified!"
    print("\n[VERIFICATION 1] Original Template Protection: PASS (Template file was not modified)")

    # 6. Verify Output File Exists and Preserves VBA & Formulas
    assert os.path.exists(output_path), "Populated output workbook does not exist!"
    wb_out = openpyxl.load_workbook(output_path, keep_vba=True, data_only=False)

    assert wb_out.vba_archive is not None, "CRITICAL ERROR: VBA Macros were stripped!"
    print("[VERIFICATION 2] VBA Macro Preservation: PASS (vbaProject.bin is preserved)")

    # 7. Check Data in BORROWERDETAILS
    ws_bd = wb_out["BORROWERDETAILS"]
    print("\n[VERIFICATION 3] BORROWERDETAILS Values:")
    print(f"  K7  (Branch)               : {ws_bd['K7'].value}")
    print(f"  K9  (Date Full)            : {ws_bd['K9'].value}")
    print(f"  K10 (Day)                  : {ws_bd['K10'].value}")
    print(f"  K11 (Month)                : {ws_bd['K11'].value}")
    print(f"  K12 (Year)                 : {ws_bd['K12'].value}")
    print(f"  K14 (Salary AC No - Text)  : {repr(ws_bd['K14'].value)} (type: {type(ws_bd['K14'].value).__name__})")
    print(f"  K16 (Loan Amount - Numeric): {ws_bd['K16'].value} (type: {type(ws_bd['K16'].value).__name__})")
    print(f"  K24 (Name)                 : {ws_bd['K24'].value}")
    print(f"  K28 (DOB)                  : {ws_bd['K28'].value}")
    print(f"  K30 (Father/Husband)       : {ws_bd['K30'].value}")
    print(f"  K39 (Mobile - Text)        : {repr(ws_bd['K39'].value)} (type: {type(ws_bd['K39'].value).__name__})")
    print(f"  K70 (PAN)                  : {ws_bd['K70'].value}")
    print(f"  K72 (Aadhaar - Text)       : {repr(ws_bd['K72'].value)} (type: {type(ws_bd['K72'].value).__name__})")
    print(f"  K75 (Gross Salary - Numeric): {ws_bd['K75'].value} (type: {type(ws_bd['K75'].value).__name__})")
    print(f"  K76 (Basic Pay - Numeric)  : {ws_bd['K76'].value} (type: {type(ws_bd['K76'].value).__name__})")
    print(f"  K80 (Gross Formula Protected): {ws_bd['K80'].value}")
    print(f"  K85 (Deductions Formula)   : {ws_bd['K85'].value}")
    print(f"  K86 (Net Salary Formula)   : {ws_bd['K86'].value}")

    assert ws_bd['K24'].value == "Subhasish Mukherjee"
    assert ws_bd['K70'].value == "ABCPM1234F"
    assert ws_bd['K72'].value == "5412 8745 9632"
    assert isinstance(ws_bd['K16'].value, (int, float)), "Financial loan amount must be numeric!"
    assert isinstance(ws_bd['K75'].value, (int, float)), "Salary must be numeric!"
    assert isinstance(ws_bd['K14'].value, str), "Account number must be text!"
    assert str(ws_bd['K80'].value).startswith("="), "Formula in K80 must remain intact!"

    # 8. Check Data in GURANTORSDETAILS
    ws_gd = wb_out["GURANTORSDETAILS"]
    print("\n[VERIFICATION 4] GURANTORSDETAILS Values:")
    print(f"  K2  (Guarantor 1 Name)      : {ws_gd['K2'].value}")
    print(f"  K3  (Guarantor 1 Father)    : {ws_gd['K3'].value}")
    print(f"  K6  (Guarantor 1 Income)    : {ws_gd['K6'].value} (type: {type(ws_gd['K6'].value).__name__})")
    print(f"  K27 (Guarantor 1 Assets Form): {ws_gd['K27'].value}")
    print(f"  K35 (Guarantor 1 Net Worth) : {ws_gd['K35'].value}")
    print(f"  K47 (Guarantor 2 Name - None): {ws_gd['K47'].value}")

    assert ws_gd['K2'].value == "Animesh Banerjee"
    assert ws_gd['K47'].value is None, "Guarantor 2 must remain blank when not provided!"
    assert str(ws_gd['K27'].value).startswith("="), "Guarantor formula must remain intact!"

    # 9. Check Data in LANDDETAILS
    ws_ld = wb_out["LANDDETAILS"]
    print("\n[VERIFICATION 5] LANDDETAILS Values:")
    print(f"  K10 (Plot / Dag No)         : {ws_ld['K10'].value}")
    print(f"  K11 (Khatian No)            : {ws_ld['K11'].value}")
    print(f"  K13 (Mouza)                 : {ws_ld['K13'].value}")
    print(f"  K26 (Deed No)               : {ws_ld['K26'].value}")
    print(f"  K42 (North Boundary)        : {ws_ld['K42'].value}")
    print(f"  K96 (Fixed Default Bank)    : {ws_ld['K96'].value}")
    print(f"  K120 (Cost Grid K)          : {ws_ld['K120'].value}")
    print(f"  Q120 (Total Formula Q)      : {ws_ld['Q120'].value}")

    assert ws_ld['K10'].value == "512"
    assert ws_ld['K26'].value == "4521/2021"
    assert ws_ld['K96'].value == "BANGIYA GRAMIN VIKASH BANK", "Fixed default must be preserved!"
    assert str(ws_ld['Q120'].value).startswith("="), "Project cost total formula must remain intact!"

    # 10. Check Downstream Formula Integrity
    print("\n[VERIFICATION 6] Downstream Form Sheets Formula Integrity:")
    app_form_f7 = wb_out["APPLICATION FORM"]["F7"].value
    app_form_d224 = wb_out["APPLICATION FORM"]["D224"].value
    ag_a12 = wb_out["AGRREMENTUNDERBHBL"]["A12"].value
    lg_a17 = wb_out["LETTER OF GURANTEE"]["A17"].value
    cersai_e16 = wb_out["CERSAI"]["E16"].value

    print(f"  APPLICATION FORM!F7     : {app_form_f7}")
    print(f"  APPLICATION FORM!D224   : {app_form_d224}")
    print(f"  AGRREMENTUNDERBHBL!A12  : {ag_a12}")
    print(f"  LETTER OF GURANTEE!A17  : {lg_a17}")
    print(f"  CERSAI!E16 (Plot No)    : {cersai_e16}")

    assert app_form_f7 == "=BORROWERDETAILS!K24"
    assert ag_a12 == "=BORROWERDETAILS!K24"
    assert lg_a17 == "=GURANTORSDETAILS!K2"
    assert cersai_e16 == "=LANDDETAILS!K10"
    print("  --> PASS: All downstream legal contract formulas remain intact and reference primary sheets.")

    print("\n" + "=" * 80)
    print("  ALL 6 MAPPING VERIFICATIONS PASSED SUCCESSFULLY!")
    print(f"  Populated Excel copy created at: {output_path}")
    print("=" * 80)

if __name__ == "__main__":
    run_excel_mapping_tests()
