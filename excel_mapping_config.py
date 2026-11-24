"""
Excel Field Mapping Configuration for WBGB Housing Loan (`WBGB HBL LOAN (2).xlsm`).
Defines the exact specifications for transferring normalized extracted data into
the primary data-entry worksheets:
1. BORROWERDETAILS
2. GURANTORSDETAILS
3. LANDDETAILS

Every mapping definition specifies:
- field_path: JSON dot-notation path in NormalizedLoanData
- sheet: Target worksheet name
- cell: Exact top-left cell coordinate
- label: Human-readable field label
- data_type: 'text', 'numeric', 'currency', 'date', 'percentage', 'integer', 'dropdown'
- mandatory: True / False
- validation_rule: Validation function or regex rule
"""

from dataclasses import dataclass
from typing import Optional, Dict, List, Any, Set, Tuple
import re

@dataclass
class FieldMapping:
    field_path: str
    sheet: str
    cell: str
    label: str
    data_type: str  # 'text', 'numeric', 'currency', 'date', 'percentage', 'integer', 'dropdown'
    mandatory: bool = False
    validation_rule: Optional[str] = None  # e.g., 'pan', 'aadhaar', 'phone', 'pin', 'date', 'positive_number'
    description: Optional[str] = None


# ==============================================================================
# PROTECTED CELLS (FORMULAS AND FIXED TEMPLATE VALUES - NEVER OVERWRITE)
# ==============================================================================

PROTECTED_CELLS: Set[Tuple[str, str]] = {
    # BORROWERDETAILS Calculated Formulas
    ("BORROWERDETAILS", "K80"),   # Gross Salary Total =K76+K77+K78+K79
    ("BORROWERDETAILS", "K85"),   # Deductions Total =K82+K83+K84
    ("BORROWERDETAILS", "K86"),   # Net Take Home Salary =K80-K85
    ("BORROWERDETAILS", "K99"),   # Total Assets =K90+K91+K92+K93+K94+K95+K96+K97+K98
    ("BORROWERDETAILS", "K107"),  # Total Liabilities =K101+K102+K103+K104+K105+K106
    ("BORROWERDETAILS", "K108"),  # Net Worth =K99-K107

    # GURANTORSDETAILS Calculated Formulas (Guarantor 1)
    ("GURANTORSDETAILS", "K27"),  # Total Assets =K23+K24+K25+K26
    ("GURANTORSDETAILS", "K34"),  # Total Liabilities =K30+K31+K32+K33
    ("GURANTORSDETAILS", "K35"),  # Net Worth =K27-K34

    # GURANTORSDETAILS Calculated Formulas (Guarantor 2)
    ("GURANTORSDETAILS", "K72"),  # Total Assets =K68+K69+K70+K71
    ("GURANTORSDETAILS", "K79"),  # Total Liabilities =K75+K76+K77+K78
    ("GURANTORSDETAILS", "K80"),  # Net Worth =K72-K79

    # LANDDETAILS Project Cost Totals
    ("LANDDETAILS", "Q120"),      # Construction Total =K120+M120+O120
    ("LANDDETAILS", "Q121"),      # Renovation Total =K121+M121+O121
    ("LANDDETAILS", "Q122"),      # Repairs Total =K122+M122+O122
    ("LANDDETAILS", "Q123"),      # Furnishing Total =K123+M123+O123

    # LANDDETAILS Fixed Template Bank Entity
    ("LANDDETAILS", "K96"),       # Fixed Default: 'BANGIYA GRAMIN VIKASH BANK'
}

# ==============================================================================
# FIELD MAPPINGS CONFIGURATION LIST
# ==============================================================================

EXCEL_FIELD_MAPPINGS: List[FieldMapping] = [
    # --------------------------------------------------------------------------
    # 1. BORROWERDETAILS (Sheet: BORROWERDETAILS)
    # --------------------------------------------------------------------------
    # Loan Application & Branch
    FieldMapping("loan.branch", "BORROWERDETAILS", "K7", "Branch Name", "text", mandatory=True),
    FieldMapping("loan.application_date", "BORROWERDETAILS", "K8", "Date of Application", "date", mandatory=False),
    FieldMapping("loan.application_date", "BORROWERDETAILS", "K9", "Application Date Full", "date", mandatory=True),
    FieldMapping("loan.application_day", "BORROWERDETAILS", "K10", "Date Format - Day", "text", mandatory=False),
    FieldMapping("loan.application_month", "BORROWERDETAILS", "K11", "Date Format - Month", "text", mandatory=False),
    FieldMapping("loan.application_year", "BORROWERDETAILS", "K12", "Date Format - Year", "integer", mandatory=False),
    FieldMapping("loan.mortgage_branch", "BORROWERDETAILS", "K13", "Mortgage Branch", "text", mandatory=False),
    FieldMapping("loan.salary_ac_no", "BORROWERDETAILS", "K14", "Salary Account Number", "text", mandatory=True, validation_rule="account_no"),
    FieldMapping("loan.loan_account_no", "BORROWERDETAILS", "K15", "Loan Account Number", "text", mandatory=False, validation_rule="account_no"),
    FieldMapping("loan.loan_amount_requested", "BORROWERDETAILS", "K16", "Loan Amount Requested", "currency", mandatory=True, validation_rule="positive_number"),
    FieldMapping("loan.term_months", "BORROWERDETAILS", "K17", "Term in Months", "integer", mandatory=False, validation_rule="positive_number"),
    FieldMapping("loan.rate_of_interest", "BORROWERDETAILS", "K18", "Rate of Interest", "percentage", mandatory=False),
    FieldMapping("loan.emi", "BORROWERDETAILS", "K19", "EMI Amount", "currency", mandatory=False, validation_rule="positive_number"),
    FieldMapping("loan.other_emi", "BORROWERDETAILS", "K20", "Other Existing EMI", "currency", mandatory=False),
    FieldMapping("loan.moratorium_period", "BORROWERDETAILS", "K21", "Moratorium Period", "text", mandatory=False),
    FieldMapping("loan.emi_starting_from", "BORROWERDETAILS", "K22", "EMI Starting From", "date", mandatory=False),
    FieldMapping("loan.purpose", "BORROWERDETAILS", "K23", "Purpose of the Loan", "text", mandatory=False),

    # Borrower Personal Details
    FieldMapping("borrower.full_name", "BORROWERDETAILS", "K24", "Name of Borrower (Full)", "text", mandatory=True),
    FieldMapping("borrower.title", "BORROWERDETAILS", "K25", "Title (Mr/Mrs/Shri)", "text", mandatory=False),
    FieldMapping("borrower.first_name", "BORROWERDETAILS", "K26", "Borrower First Name", "text", mandatory=False),
    FieldMapping("borrower.surname", "BORROWERDETAILS", "K27", "Borrower Surname", "text", mandatory=False),
    FieldMapping("borrower.dob", "BORROWERDETAILS", "K28", "Date of Birth", "date", mandatory=True, validation_rule="date"),
    FieldMapping("borrower.gender", "BORROWERDETAILS", "K29", "Gender / Sex", "dropdown", mandatory=False),
    FieldMapping("borrower.father_husband_name", "BORROWERDETAILS", "K30", "Father's / Husband's Name", "text", mandatory=True),
    FieldMapping("borrower.qualification", "BORROWERDETAILS", "K31", "Educational Qualification", "text", mandatory=False),
    FieldMapping("borrower.marital_status", "BORROWERDETAILS", "K32", "Marital Status", "dropdown", mandatory=False),
    FieldMapping("borrower.num_dependents", "BORROWERDETAILS", "K33", "Number of Dependents", "integer", mandatory=False),

    # Residential Address
    FieldMapping("borrower.resident_address.vill", "BORROWERDETAILS", "K35", "Residential Village / Street", "text", mandatory=False),
    FieldMapping("borrower.resident_address.post", "BORROWERDETAILS", "K36", "Residential Post Office", "text", mandatory=False),
    FieldMapping("borrower.resident_address.dist", "BORROWERDETAILS", "K37", "Residential District", "text", mandatory=False),
    FieldMapping("borrower.resident_address.pin", "BORROWERDETAILS", "K38", "Residential PIN Code", "text", mandatory=False, validation_rule="pin"),
    FieldMapping("borrower.mobile_no", "BORROWERDETAILS", "K39", "Mobile Number", "text", mandatory=True, validation_rule="phone"),

    # Present Posting Address
    FieldMapping("borrower.present_posting.org_name", "BORROWERDETAILS", "K41", "Posting Organization Name", "text", mandatory=False),
    FieldMapping("borrower.present_posting.vill", "BORROWERDETAILS", "K42", "Posting Village", "text", mandatory=False),
    FieldMapping("borrower.present_posting.post", "BORROWERDETAILS", "K43", "Posting Post Office", "text", mandatory=False),
    FieldMapping("borrower.present_posting.dist", "BORROWERDETAILS", "K44", "Posting District", "text", mandatory=False),
    FieldMapping("borrower.present_posting.pin", "BORROWERDETAILS", "K45", "Posting PIN Code", "text", mandatory=False, validation_rule="pin"),

    # Employer Address
    FieldMapping("borrower.employer.employer_name", "BORROWERDETAILS", "K47", "Employer Name", "text", mandatory=False),
    FieldMapping("borrower.employer.vill", "BORROWERDETAILS", "K48", "Employer Village", "text", mandatory=False),
    FieldMapping("borrower.employer.post", "BORROWERDETAILS", "K49", "Employer Post Office", "text", mandatory=False),
    FieldMapping("borrower.employer.dist", "BORROWERDETAILS", "K50", "Employer District", "text", mandatory=False),
    FieldMapping("borrower.employer.pin", "BORROWERDETAILS", "K51", "Employer PIN Code", "text", mandatory=False, validation_rule="pin"),

    # Service Details
    FieldMapping("borrower.service.length_of_service", "BORROWERDETAILS", "K52", "Length of Service", "text", mandatory=False),
    FieldMapping("borrower.service.designation", "BORROWERDETAILS", "K53", "Applicant Designation", "text", mandatory=False),
    FieldMapping("borrower.service.date_of_joining", "BORROWERDETAILS", "K54", "Date of Joining", "date", mandatory=False, validation_rule="date"),
    FieldMapping("borrower.service.date_of_retirement", "BORROWERDETAILS", "K55", "Date of Retirement", "date", mandatory=False, validation_rule="date"),

    # Personal Reference 1
    FieldMapping("references[0].name", "BORROWERDETAILS", "K57", "Reference 1 Name", "text", mandatory=False),
    FieldMapping("references[0].vill", "BORROWERDETAILS", "K58", "Reference 1 Village", "text", mandatory=False),
    FieldMapping("references[0].post", "BORROWERDETAILS", "K59", "Reference 1 Post", "text", mandatory=False),
    FieldMapping("references[0].dist", "BORROWERDETAILS", "K60", "Reference 1 District", "text", mandatory=False),
    FieldMapping("references[0].pin", "BORROWERDETAILS", "K61", "Reference 1 PIN", "text", mandatory=False, validation_rule="pin"),
    FieldMapping("references[0].mobile_no", "BORROWERDETAILS", "K62", "Reference 1 Mobile No", "text", mandatory=False, validation_rule="phone"),

    # Personal Reference 2
    FieldMapping("references[1].name", "BORROWERDETAILS", "K63", "Reference 2 Name", "text", mandatory=False),
    FieldMapping("references[1].vill", "BORROWERDETAILS", "K64", "Reference 2 Village", "text", mandatory=False),
    FieldMapping("references[1].post", "BORROWERDETAILS", "K65", "Reference 2 Post", "text", mandatory=False),
    FieldMapping("references[1].dist", "BORROWERDETAILS", "K66", "Reference 2 District", "text", mandatory=False),
    FieldMapping("references[1].pin", "BORROWERDETAILS", "K67", "Reference 2 PIN", "text", mandatory=False, validation_rule="pin"),
    FieldMapping("references[1].mobile_no", "BORROWERDETAILS", "K68", "Reference 2 Mobile No", "text", mandatory=False, validation_rule="phone"),

    # KYC Details of Borrower
    FieldMapping("borrower.pan_no", "BORROWERDETAILS", "K70", "PAN Number", "text", mandatory=True, validation_rule="pan"),
    FieldMapping("borrower.voter_card_no", "BORROWERDETAILS", "K71", "Voter Card Number", "text", mandatory=False),
    FieldMapping("borrower.aadhar_no", "BORROWERDETAILS", "K72", "Aadhaar Number", "text", mandatory=True, validation_rule="aadhaar"),

    # Salary Details of Borrower
    FieldMapping("salary.salary_month", "BORROWERDETAILS", "K74", "Salary Month", "text", mandatory=False),
    FieldMapping("salary.gross_salary", "BORROWERDETAILS", "K75", "Gross Salary", "currency", mandatory=True, validation_rule="positive_number"),
    FieldMapping("salary.basic_pay", "BORROWERDETAILS", "K76", "Basic Pay", "currency", mandatory=True, validation_rule="positive_number"),
    FieldMapping("salary.da", "BORROWERDETAILS", "K77", "Dearness Allowance (D.A)", "currency", mandatory=False, validation_rule="positive_number"),
    FieldMapping("salary.hra", "BORROWERDETAILS", "K78", "House Rent Allowance (HRA)", "currency", mandatory=False, validation_rule="positive_number"),
    FieldMapping("salary.others_allowance", "BORROWERDETAILS", "K79", "Other Allowances", "currency", mandatory=False, validation_rule="positive_number"),
    # K80 is Gross Salary Formula =K76+K77+K78+K79
    FieldMapping("salary.deductions.pf", "BORROWERDETAILS", "K82", "Provident Fund (P.F)", "currency", mandatory=False, validation_rule="positive_number"),
    FieldMapping("salary.deductions.it", "BORROWERDETAILS", "K83", "Income Tax (I.T)", "currency", mandatory=False, validation_rule="positive_number"),
    FieldMapping("salary.deductions.others", "BORROWERDETAILS", "K84", "Other Deductions", "currency", mandatory=False, validation_rule="positive_number"),
    # K85 is Deduction Total Formula =K82+K83+K84
    # K86 is Net Take Home Formula =K80-K85
    FieldMapping("salary.family_expenditure", "BORROWERDETAILS", "K87", "Family Expenditure", "currency", mandatory=False, validation_rule="positive_number"),
    FieldMapping("salary.p_tax_last_paid", "BORROWERDETAILS", "K88", "P. Tax Last Paid", "currency", mandatory=False, validation_rule="positive_number"),

    # Assets of Borrower
    FieldMapping("assets.bank_deposit", "BORROWERDETAILS", "K90", "Bank & Other Deposit", "currency", mandatory=False),
    FieldMapping("assets.life_insurance", "BORROWERDETAILS", "K91", "Life Insurance Policies", "currency", mandatory=False),
    FieldMapping("assets.other_movable", "BORROWERDETAILS", "K92", "Other Movable Property", "currency", mandatory=False),
    FieldMapping("assets.immovable_house", "BORROWERDETAILS", "K96", "Immovable Property - House", "currency", mandatory=False),
    FieldMapping("assets.immovable_land", "BORROWERDETAILS", "K97", "Immovable Property - Land", "currency", mandatory=False),
    # K99 is Total Assets Formula

    # Liabilities of Borrower
    FieldMapping("liabilities.from_employer", "BORROWERDETAILS", "K101", "Liability From Employer", "currency", mandatory=False),
    FieldMapping("liabilities.from_family_friends", "BORROWERDETAILS", "K102", "Liability From Family & Friends", "currency", mandatory=False),
    FieldMapping("liabilities.from_provident_fund", "BORROWERDETAILS", "K103", "Liability From Provident Fund", "currency", mandatory=False),
    FieldMapping("liabilities.from_other_source", "BORROWERDETAILS", "K104", "Liability From Other Sources", "currency", mandatory=False),
    # K107 is Total Liabilities Formula, K108 is Net Worth Formula
    FieldMapping("salary.form16_last_submitted", "BORROWERDETAILS", "K109", "Form 16 Last Submitted Year", "text", mandatory=False),

    # Dependent Family Members of Borrower
    FieldMapping("dependents[0].name", "BORROWERDETAILS", "K111", "Dependent 1 Name", "text", mandatory=False),
    FieldMapping("dependents[0].age", "BORROWERDETAILS", "K112", "Dependent 1 Age", "integer", mandatory=False),
    FieldMapping("dependents[0].relationship", "BORROWERDETAILS", "K113", "Dependent 1 Relationship", "dropdown", mandatory=False),
    FieldMapping("dependents[1].name", "BORROWERDETAILS", "K114", "Dependent 2 Name", "text", mandatory=False),
    FieldMapping("dependents[1].age", "BORROWERDETAILS", "K115", "Dependent 2 Age", "integer", mandatory=False),
    FieldMapping("dependents[1].relationship", "BORROWERDETAILS", "K116", "Dependent 2 Relationship", "dropdown", mandatory=False),
    FieldMapping("dependents[2].name", "BORROWERDETAILS", "K117", "Dependent 3 Name", "text", mandatory=False),
    FieldMapping("dependents[2].age", "BORROWERDETAILS", "K118", "Dependent 3 Age", "integer", mandatory=False),
    FieldMapping("dependents[2].relationship", "BORROWERDETAILS", "K119", "Dependent 3 Relationship", "dropdown", mandatory=False),

    # --------------------------------------------------------------------------
    # 2. GURANTORSDETAILS (Sheet: GURANTORSDETAILS)
    # --------------------------------------------------------------------------
    # --- Guarantor 1 (Rows 1 to 45) ---
    FieldMapping("guarantors[0].name", "GURANTORSDETAILS", "K2", "Guarantor 1 Name", "text", mandatory=False),
    FieldMapping("guarantors[0].father_husband_name", "GURANTORSDETAILS", "K3", "Guarantor 1 Father/Husband", "text", mandatory=False),
    FieldMapping("guarantors[0].dob", "GURANTORSDETAILS", "K4", "Guarantor 1 Date of Birth", "date", mandatory=False, validation_rule="date"),
    FieldMapping("guarantors[0].occupation", "GURANTORSDETAILS", "K5", "Guarantor 1 Present Occupation", "text", mandatory=False),
    FieldMapping("guarantors[0].monthly_income", "GURANTORSDETAILS", "K6", "Guarantor 1 Monthly Income", "currency", mandatory=False, validation_rule="positive_number"),
    FieldMapping("guarantors[0].resident_address.vill", "GURANTORSDETAILS", "K8", "Guarantor 1 Resident Vill", "text", mandatory=False),
    FieldMapping("guarantors[0].resident_address.post", "GURANTORSDETAILS", "K9", "Guarantor 1 Resident Post", "text", mandatory=False),
    FieldMapping("guarantors[0].resident_address.dist", "GURANTORSDETAILS", "K10", "Guarantor 1 Resident Dist", "text", mandatory=False),
    FieldMapping("guarantors[0].resident_address.pin", "GURANTORSDETAILS", "K11", "Guarantor 1 Resident PIN", "text", mandatory=False, validation_rule="pin"),
    FieldMapping("guarantors[0].p_tax_paid_upto", "GURANTORSDETAILS", "K12", "Guarantor 1 P Tax Paid Upto", "text", mandatory=False),
    FieldMapping("guarantors[0].mobile_no", "GURANTORSDETAILS", "K13", "Guarantor 1 Mobile No", "text", mandatory=False, validation_rule="phone"),
    FieldMapping("guarantors[0].work_address.vill", "GURANTORSDETAILS", "K15", "Guarantor 1 Work Vill", "text", mandatory=False),
    FieldMapping("guarantors[0].work_address.post", "GURANTORSDETAILS", "K16", "Guarantor 1 Work Post", "text", mandatory=False),
    FieldMapping("guarantors[0].work_address.dist", "GURANTORSDETAILS", "K17", "Guarantor 1 Work Dist", "text", mandatory=False),
    FieldMapping("guarantors[0].work_address.pin", "GURANTORSDETAILS", "K18", "Guarantor 1 Work PIN", "text", mandatory=False, validation_rule="pin"),
    FieldMapping("guarantors[0].qualification", "GURANTORSDETAILS", "K19", "Guarantor 1 Educational Qualification", "text", mandatory=False),
    FieldMapping("guarantors[0].account_no", "GURANTORSDETAILS", "K20", "Guarantor 1 Account Number", "text", mandatory=False, validation_rule="account_no"),
    FieldMapping("guarantors[0].assets[0].amount", "GURANTORSDETAILS", "K23", "Guarantor 1 Asset 1 Amount", "currency", mandatory=False),
    FieldMapping("guarantors[0].assets[1].amount", "GURANTORSDETAILS", "K24", "Guarantor 1 Asset 2 Amount", "currency", mandatory=False),
    FieldMapping("guarantors[0].assets[2].amount", "GURANTORSDETAILS", "K25", "Guarantor 1 Asset 3 Amount", "currency", mandatory=False),
    FieldMapping("guarantors[0].assets[3].amount", "GURANTORSDETAILS", "K26", "Guarantor 1 Asset 4 Amount", "currency", mandatory=False),
    # K27 is Guarantor 1 Total Assets Formula =K23+K24+K25+K26
    FieldMapping("guarantors[0].liabilities[0].amount", "GURANTORSDETAILS", "K30", "Guarantor 1 Liability 1 Amount", "currency", mandatory=False),
    FieldMapping("guarantors[0].liabilities[1].amount", "GURANTORSDETAILS", "K31", "Guarantor 1 Liability 2 Amount", "currency", mandatory=False),
    FieldMapping("guarantors[0].liabilities[2].amount", "GURANTORSDETAILS", "K32", "Guarantor 1 Liability 3 Amount", "currency", mandatory=False),
    FieldMapping("guarantors[0].liabilities[3].amount", "GURANTORSDETAILS", "K33", "Guarantor 1 Liability 4 Amount", "currency", mandatory=False),
    # K34 is Guarantor 1 Total Liabilities Formula, K35 is Net Worth Formula

    # --- Guarantor 2 (Rows 46 to 90 - Conditional if present) ---
    FieldMapping("guarantors[1].name", "GURANTORSDETAILS", "K47", "Guarantor 2 Name", "text", mandatory=False),
    FieldMapping("guarantors[1].father_husband_name", "GURANTORSDETAILS", "K48", "Guarantor 2 Father/Husband", "text", mandatory=False),
    FieldMapping("guarantors[1].dob", "GURANTORSDETAILS", "K49", "Guarantor 2 Date of Birth", "date", mandatory=False, validation_rule="date"),
    FieldMapping("guarantors[1].occupation", "GURANTORSDETAILS", "K50", "Guarantor 2 Present Occupation", "text", mandatory=False),
    FieldMapping("guarantors[1].monthly_income", "GURANTORSDETAILS", "K51", "Guarantor 2 Monthly Income", "currency", mandatory=False, validation_rule="positive_number"),
    FieldMapping("guarantors[1].resident_address.vill", "GURANTORSDETAILS", "K53", "Guarantor 2 Resident Vill", "text", mandatory=False),
    FieldMapping("guarantors[1].resident_address.post", "GURANTORSDETAILS", "K54", "Guarantor 2 Resident Post", "text", mandatory=False),
    FieldMapping("guarantors[1].resident_address.dist", "GURANTORSDETAILS", "K55", "Guarantor 2 Resident Dist", "text", mandatory=False),
    FieldMapping("guarantors[1].resident_address.pin", "GURANTORSDETAILS", "K56", "Guarantor 2 Resident PIN", "text", mandatory=False, validation_rule="pin"),
    FieldMapping("guarantors[1].p_tax_paid_upto", "GURANTORSDETAILS", "K57", "Guarantor 2 P Tax Paid Upto", "text", mandatory=False),
    FieldMapping("guarantors[1].mobile_no", "GURANTORSDETAILS", "K58", "Guarantor 2 Mobile No", "text", mandatory=False, validation_rule="phone"),
    FieldMapping("guarantors[1].work_address.vill", "GURANTORSDETAILS", "K60", "Guarantor 2 Work Vill", "text", mandatory=False),
    FieldMapping("guarantors[1].work_address.post", "GURANTORSDETAILS", "K61", "Guarantor 2 Work Post", "text", mandatory=False),
    FieldMapping("guarantors[1].work_address.dist", "GURANTORSDETAILS", "K62", "Guarantor 2 Work Dist", "text", mandatory=False),
    FieldMapping("guarantors[1].work_address.pin", "GURANTORSDETAILS", "K63", "Guarantor 2 Work PIN", "text", mandatory=False, validation_rule="pin"),
    FieldMapping("guarantors[1].qualification", "GURANTORSDETAILS", "K64", "Guarantor 2 Educational Qualification", "text", mandatory=False),
    FieldMapping("guarantors[1].account_no", "GURANTORSDETAILS", "K65", "Guarantor 2 Account Number", "text", mandatory=False, validation_rule="account_no"),

    # --------------------------------------------------------------------------
    # 3. LANDDETAILS (Sheet: LANDDETAILS)
    # --------------------------------------------------------------------------
    FieldMapping("property.inspection_date", "LANDDETAILS", "K2", "Date of Inspection", "date", mandatory=False),
    FieldMapping("property.mortgage_branch", "LANDDETAILS", "K3", "Mortgage Branch", "text", mandatory=False),
    FieldMapping("property.sub_registry_office", "LANDDETAILS", "K4", "Sub Registry Office", "text", mandatory=False),
    FieldMapping("property.plan_approved_by", "LANDDETAILS", "K5", "Plan Approved By", "text", mandatory=False),
    FieldMapping("property.flat_no", "LANDDETAILS", "K6", "Flat Number", "text", mandatory=False),
    FieldMapping("property.house_no", "LANDDETAILS", "K7", "House Number", "text", mandatory=False),
    FieldMapping("property.premises_no", "LANDDETAILS", "K8", "Premises Number", "text", mandatory=False),
    FieldMapping("property.building_name", "LANDDETAILS", "K9", "Name of the Building", "text", mandatory=False),
    FieldMapping("property.plot_dag_no", "LANDDETAILS", "K10", "Plot / Dag Number", "text", mandatory=True),
    FieldMapping("property.khatian_no", "LANDDETAILS", "K11", "Khatian Number", "text", mandatory=True),
    FieldMapping("property.jl_no", "LANDDETAILS", "K12", "J.L. Number", "text", mandatory=True),
    FieldMapping("property.mouza", "LANDDETAILS", "K13", "Mouza", "text", mandatory=True),
    FieldMapping("property.block", "LANDDETAILS", "K14", "Block", "text", mandatory=False),
    FieldMapping("property.ps", "LANDDETAILS", "K15", "Police Station (P.S)", "text", mandatory=False),
    FieldMapping("property.post", "LANDDETAILS", "K16", "Property Post Office", "text", mandatory=False),
    FieldMapping("property.pin", "LANDDETAILS", "K17", "Property PIN Code", "text", mandatory=False, validation_rule="pin"),
    FieldMapping("property.dist", "LANDDETAILS", "K18", "Property District", "text", mandatory=False),
    FieldMapping("property.area", "LANDDETAILS", "K19", "Land Area", "text", mandatory=True),
    FieldMapping("property.classification", "LANDDETAILS", "K20", "Land Classification (Nature)", "text", mandatory=False),
    FieldMapping("property.proposed_buildup_area", "LANDDETAILS", "K21", "Proposed Buildup Area", "text", mandatory=False),
    FieldMapping("property.purchase_price", "LANDDETAILS", "K22", "Purchase Price of Land", "currency", mandatory=False, validation_rule="positive_number"),
    FieldMapping("property.purchase_date", "LANDDETAILS", "K23", "Date of Purchase", "date", mandatory=False, validation_rule="date"),
    FieldMapping("property.municipality_gp", "LANDDETAILS", "K24", "Municipality / GP", "text", mandatory=False),
    FieldMapping("property.street_name", "LANDDETAILS", "K25", "Street / Road Name", "text", mandatory=False),
    FieldMapping("property.deed_no", "LANDDETAILS", "K26", "Title Deed Number", "text", mandatory=True),
    FieldMapping("property.book_no", "LANDDETAILS", "K27", "Book Number", "text", mandatory=False),
    FieldMapping("property.vol_no", "LANDDETAILS", "K28", "Volume Number", "text", mandatory=False),
    FieldMapping("property.page_no", "LANDDETAILS", "K29", "Page Number", "text", mandatory=False),
    FieldMapping("property.area", "LANDDETAILS", "K30", "Covering Land", "text", mandatory=False),
    FieldMapping("property.survey_no", "LANDDETAILS", "K31", "Survey Number", "text", mandatory=False),
    FieldMapping("property.landmark", "LANDDETAILS", "K32", "Landmark", "text", mandatory=False),
    FieldMapping("property.latitude_north", "LANDDETAILS", "K33", "Latitude (North)", "text", mandatory=False),
    FieldMapping("property.latitude_east", "LANDDETAILS", "K34", "Latitude (East)", "text", mandatory=False),
    FieldMapping("property.taluka", "LANDDETAILS", "K35", "Taluka / Tehsil", "text", mandatory=False),

    # Boundaries of Property
    FieldMapping("property.boundaries.north", "LANDDETAILS", "K42", "Boundary North", "text", mandatory=True),
    FieldMapping("property.boundaries.south", "LANDDETAILS", "K43", "Boundary South", "text", mandatory=True),
    FieldMapping("property.boundaries.east", "LANDDETAILS", "K44", "Boundary East", "text", mandatory=True),
    FieldMapping("property.boundaries.west", "LANDDETAILS", "K45", "Boundary West", "text", mandatory=True),

    # Property Valuations
    FieldMapping("property.valuation_independent", "LANDDETAILS", "K53", "Market Value (Independent Enquiry)", "currency", mandatory=False, validation_rule="positive_number"),
    FieldMapping("property.valuation_empanelled", "LANDDETAILS", "K54", "Market Value (Empanelled Valuer)", "currency", mandatory=False, validation_rule="positive_number"),

    # Legal Searching & Valuation Details
    FieldMapping("property.search_by", "LANDDETAILS", "K80", "Searching Made By", "text", mandatory=False),
    FieldMapping("property.search_report_date", "LANDDETAILS", "K81", "Date of Searching Report", "date", mandatory=False, validation_rule="date"),
    FieldMapping("property.valuation_date", "LANDDETAILS", "K84", "Date of Valuation", "date", mandatory=False, validation_rule="date"),
    FieldMapping("property.valuer_name", "LANDDETAILS", "K85", "Valuation Done By", "text", mandatory=False),

    # Security Interest & CERSAI
    # K96 is Fixed: 'BANGIYA GRAMIN VIKASH BANK'
    FieldMapping("property.bsr_code", "LANDDETAILS", "K97", "BSR Code", "text", mandatory=False),
    FieldMapping("property.security_interest_creation_date", "LANDDETAILS", "K110", "Security Interest Creation Date", "date", mandatory=False, validation_rule="date"),
    FieldMapping("property.document_type", "LANDDETAILS", "K111", "Document Type", "text", mandatory=False),
    FieldMapping("property.deed_no", "LANDDETAILS", "K112", "Title Document No", "text", mandatory=False),
    FieldMapping("property.document_registration_date", "LANDDETAILS", "K113", "Document / Registration Date", "date", mandatory=False, validation_rule="date"),

    # Multi-Column Project Cost Grid (Rows 119 to 123)
    # Construction Cost Breakdown
    FieldMapping("property.project_cost.construction_cost", "LANDDETAILS", "K120", "Construction Cost (1. Cost)", "currency", mandatory=False, validation_rule="positive_number"),
    FieldMapping("property.project_cost.construction_incidental", "LANDDETAILS", "M120", "Construction Incidental (2. Incidental)", "currency", mandatory=False, validation_rule="positive_number"),
    FieldMapping("property.project_cost.construction_contingency", "LANDDETAILS", "O120", "Construction Contingency (3. Contingency)", "currency", mandatory=False, validation_rule="positive_number"),
    # Q120 is Formula =K120+M120+O120

    # Renovation Cost Breakdown
    FieldMapping("property.project_cost.renovation_cost", "LANDDETAILS", "K121", "Renovation Cost", "currency", mandatory=False, validation_rule="positive_number"),
    # Repairs Cost Breakdown
    FieldMapping("property.project_cost.repairs_cost", "LANDDETAILS", "K122", "Repairs Cost", "currency", mandatory=False, validation_rule="positive_number"),
    # Furnishing Cost Breakdown
    FieldMapping("property.project_cost.furnishing_cost", "LANDDETAILS", "K123", "Furnishing / Extension Cost", "currency", mandatory=False, validation_rule="positive_number"),
]


# ==============================================================================
# VALIDATION RULES REGISTRY
# ==============================================================================

VALIDATION_RULES = {
    "pan": lambda val: bool(re.match(r'^[A-Z]{5}[0-9]{4}[A-Z]$', str(val).strip().upper())),
    "aadhaar": lambda val: len(re.sub(r'\D', '', str(val))) == 12,
    "phone": lambda val: len(re.sub(r'\D', '', str(val))) in (10, 11, 12),
    "pin": lambda val: len(re.sub(r'\D', '', str(val))) == 6,
    "account_no": lambda val: len(re.sub(r'\D', '', str(val))) >= 8,
    "positive_number": lambda val: _is_positive_number(val),
    "date": lambda val: _is_valid_date(val)
}

def _is_positive_number(val: Any) -> bool:
    try:
        clean = re.sub(r'[^\d.]', '', str(val))
        return float(clean) >= 0
    except (ValueError, TypeError):
        return False

def _is_valid_date(val: Any) -> bool:
    s = str(val).strip()
    return bool(re.search(r'\d{1,2}[\/\.\-]\d{1,2}[\/\.\-]\d{2,4}', s))
