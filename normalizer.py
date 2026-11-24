"""
Normalization and Central Data Structure Module for WBGB Housing Loan Application.
Translates heterogeneous multi-document extraction JSON from OCR/LLM pipelines
into a single, validated, normalized structure ready for Excel mapping.

Key Guarantees:
1. Pure segregation: Borrower, Guarantor 1, and Guarantor 2 are NEVER merged.
2. Complete provenance: Every field stores source_document, confidence, evidence, and history.
3. Conflict tracking: Multiple extractions of the same field retain all variants and flag conflicts.
4. Non-hallucination: Missing fields are strictly None/null, never invented.
5. Bidirectional: Provides structured nested access and flat Excel-coordinate ready dictionaries.
"""

import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

# ==============================================================================
# FIELD DATA CONTAINER
# ==============================================================================

class NormalizedField:
    """
    Represents a single extracted data point with full audit trace.
    Preserves original extracted value, confidence, evidence snippet,
    source document name, and conflict history.
    """
    def __init__(
        self,
        key: str,
        value: Optional[Any] = None,
        source_document: Optional[str] = None,
        confidence: Optional[float] = None,
        evidence: Optional[str] = None,
        raw_value: Optional[Any] = None,
    ):
        self.key = key
        self.value = value
        self.raw_value = raw_value if raw_value is not None else value
        self.source_document = source_document
        self.confidence = confidence
        self.evidence = evidence
        self.history: List[Dict[str, Any]] = []
        self.has_conflict: bool = False
        self.is_valid: bool = True
        self.validation_note: Optional[str] = None

        if value is not None:
            self.history.append({
                "value": self.value,
                "raw_value": self.raw_value,
                "source_document": self.source_document,
                "confidence": self.confidence,
                "evidence": self.evidence,
                "timestamp": datetime.now().isoformat()
            })

    def update(
        self,
        new_value: Any,
        source_document: Optional[str] = None,
        confidence: Optional[float] = None,
        evidence: Optional[str] = None,
        raw_value: Optional[Any] = None,
    ):
        """Adds a new extracted candidate. Tracks conflicts if values differ."""
        if new_value is None or str(new_value).strip() == "":
            return

        new_raw = raw_value if raw_value is not None else new_value
        entry = {
            "value": new_value,
            "raw_value": new_raw,
            "source_document": source_document,
            "confidence": confidence,
            "evidence": evidence,
            "timestamp": datetime.now().isoformat()
        }

        # If field is currently empty, initialize it
        if self.value is None or str(self.value).strip() == "":
            self.value = new_value
            self.raw_value = new_raw
            self.source_document = source_document
            self.confidence = confidence
            self.evidence = evidence
            self.history.append(entry)
            return

        # If field already has a value, check for conflict
        self.history.append(entry)
        curr_str = str(self.value).strip().lower()
        new_str = str(new_value).strip().lower()

        if curr_str != new_str:
            self.has_conflict = True
            # Prefer higher confidence if provided
            if confidence is not None and (self.confidence is None or confidence > self.confidence):
                self.value = new_value
                self.raw_value = new_raw
                self.source_document = source_document
                self.confidence = confidence
                self.evidence = evidence
            # If current value was short or truncated and new value is fuller, update primary
            elif len(new_str) > len(curr_str) and (self.confidence is None or confidence == self.confidence):
                self.value = new_value
                self.raw_value = new_raw
                self.source_document = source_document
                self.confidence = confidence
                self.evidence = evidence

    def to_dict(self) -> Dict[str, Any]:
        """Serializes field to JSON-friendly dictionary."""
        return {
            "value": self.value,
            "raw_value": self.raw_value,
            "source_document": self.source_document,
            "confidence": self.confidence,
            "evidence": self.evidence,
            "has_conflict": self.has_conflict,
            "is_valid": self.is_valid,
            "validation_note": self.validation_note,
            "history": self.history
        }


# ==============================================================================
# FIELD ALIASES AND MAPPING RULES
# ==============================================================================

# Common label variations extracted across Indian documents, OCR engines, and LLMs
LABEL_ALIASES = {
    # Borrower Personal / KYC
    "borrower.full_name": [
        "full name", "borrower name", "applicant name", "name of applicant",
        "holder's name", "name of holder", "name", "customer name", "purchaser / buyer"
    ],
    "borrower.father_husband_name": [
        "father's name", "fathers name", "father/husband name", "father / husband name",
        "care of", "c/o", "husband name", "father name"
    ],
    "borrower.dob": [
        "date of birth", "dob", "birth date", "born on"
    ],
    "borrower.gender": [
        "gender", "sex"
    ],
    "borrower.marital_status": [
        "marital status", "marital_status"
    ],
    "borrower.qualification": [
        "qualification", "educational qualification", "education"
    ],
    "borrower.num_dependents": [
        "no of dependent", "number of dependents", "no of dependents", "dependents"
    ],
    "borrower.mobile_no": [
        "mobile no", "mobile number", "mobile", "phone no", "phone number", "contact no"
    ],
    "borrower.pan_no": [
        "pan no", "pan number", "pan", "permanent account number"
    ],
    "borrower.aadhar_no": [
        "aadhar no", "aadhar number", "aadhaar no", "aadhaar number", "aadhaar", "uid"
    ],
    "borrower.voter_card_no": [
        "voter card no", "voter id", "voter id no", "epic no", "epic"
    ],
    "borrower.driving_license_no": [
        "driving license number", "driving licence no", "dl number", "dl no", "license number"
    ],

    # Residential Address
    "borrower.resident_address.vill": ["resident vill", "village", "vill", "address vill", "premises"],
    "borrower.resident_address.post": ["resident post", "post office", "post", "po"],
    "borrower.resident_address.dist": ["resident dist", "district", "dist"],
    "borrower.resident_address.pin": ["resident pin", "pin code", "pincode", "pin"],

    # Employment & Posting
    "borrower.present_posting.org_name": ["name of the organization", "organization name", "organization", "office name"],
    "borrower.present_posting.vill": ["present posting vill", "posting vill"],
    "borrower.present_posting.post": ["present posting post", "posting post"],
    "borrower.present_posting.dist": ["present posting dist", "posting dist"],
    "borrower.present_posting.pin": ["present posting pin", "posting pin"],

    # Employer Address
    "borrower.employer.employer_name": ["name of the employer", "employer name", "company name"],
    "borrower.employer.vill": ["employer vill"],
    "borrower.employer.post": ["employer post"],
    "borrower.employer.dist": ["employer dist"],
    "borrower.employer.pin": ["employer pin"],

    # Service Details
    "borrower.service.length_of_service": ["length of service", "total service", "experience"],
    "borrower.service.designation": ["applicant designation", "designation", "job title", "role"],
    "borrower.service.date_of_joining": ["date of joining", "doj", "joining date"],
    "borrower.service.date_of_retirement": ["date of retirement", "dor", "retirement date"],

    # Guarantor Specific Fields
    "guarantor.monthly_income": ["monthly income", "guarantor income", "income"],
    "guarantor.occupation": ["present occupation", "occupation"],
    "guarantor.p_tax_paid_upto": ["p tax paid upto", "guarantor p tax"],
    "guarantor.account_no": ["guarantor account no", "guarantor a/c no"],

    # Salary Details
    "salary.salary_month": ["details of salary for the month of", "salary month", "payslip month", "month"],
    "salary.gross_salary": ["gross salary", "gross pay", "gross earnings", "total earnings"],
    "salary.basic_pay": ["basic pay", "basic", "basic salary"],
    "salary.da": ["d.a", "da", "dearness allowance"],
    "salary.hra": ["hra", "house rent allowance"],
    "salary.others_allowance": ["salary others", "other allowance", "special allowance"],
    "salary.deduction_pf": ["p.f", "pf", "provident fund", "gpf", "epf"],
    "salary.deduction_it": ["i.t", "it", "income tax", "tds"],
    "salary.deduction_others": ["deductions others", "other deductions", "prof tax deduction"],
    "salary.family_expenditure": ["family expenditure", "monthly expenditure", "family expense"],
    "salary.p_tax_last_paid": ["p tax last paid", "professional tax", "p tax"],
    "salary.form16_last_submitted": ["form 16 last submited", "form 16", "form 16 submitted"],

    # Loan Request Details
    "loan.branch": ["branch", "branch name", "home branch", "applying branch"],
    "loan.application_date": ["date of application", "application date", "loan application date", "date"],
    "loan.mortgage_branch": ["mortgage branch"],
    "loan.salary_ac_no": ["salary ac no", "salary account no", "salary account number"],
    "loan.loan_account_no": ["loan account no", "loan account number", "loan ac no"],
    "loan.loan_amount_requested": ["loan requested", "loan amount requested", "required loan", "loan amount"],
    "loan.term_months": ["term of the loan in month", "term of loan", "tenure months", "tenure in months", "tenure"],
    "loan.rate_of_interest": ["rate of interest", "interest rate", "roi"],
    "loan.emi": ["emi", "monthly emi", "installment"],
    "loan.other_emi": ["if any other emi", "other emi", "existing emi"],
    "loan.moratorium_period": ["moratorium period", "moratorium"],
    "loan.emi_starting_from": ["emi starting from", "emi start date"],
    "loan.purpose": ["purpose of the loan", "loan purpose", "purpose"],

    # Property / Land Details
    "property.inspection_date": ["date of inspection", "inspection date"],
    "property.mortgage_branch": ["land mortgage branch", "property mortgage branch"],
    "property.sub_registry_office": ["sub registry office", "sro", "registry office", "sub-registry office"],
    "property.plan_approved_by": ["plan approved by", "sanction authority", "approval authority"],
    "property.flat_no": ["flat no", "flat number", "unit no"],
    "property.house_no": ["house no", "house number", "holding no"],
    "property.premises_no": ["premises no", "premises number"],
    "property.building_name": ["name of the building", "building name", "apartment name"],
    "property.plot_dag_no": ["plot/ dag no", "plot no", "dag no", "plot / dag no", "rs dag", "lr dag"],
    "property.khatian_no": ["kaitian no", "khatian no", "rs khatian", "lr khatian"],
    "property.jl_no": ["jl no", "j.l. no", "j.l. no."],
    "property.mouza": ["mouza", "mauza"],
    "property.block": ["block", "c.d. block", "cd block"],
    "property.ps": ["p.s", "ps", "police station"],
    "property.post": ["property post", "land post"],
    "property.pin": ["property pin", "land pin"],
    "property.dist": ["property dist", "land dist"],
    "property.area": ["area", "land area", "property area", "cvering land"],
    "property.classification": ["classification", "land classification", "nature of land"],
    "property.proposed_buildup_area": ["proposed builup area", "proposed buildup area", "buildup area"],
    "property.purchase_price": ["purchase price of land", "consideration price", "purchase price", "cost of land rs."],
    "property.purchase_date": ["date of purchase", "purchase date"],
    "property.municipality_gp": ["municipality/gp", "gram panchayat", "municipality", "panchayat"],
    "property.street_name": ["name of the street", "street name", "road name"],
    "property.deed_no": ["deed no", "deed number", "title deed no", "tittle doccument no"],
    "property.book_no": ["book no", "book number"],
    "property.vol_no": ["vol no", "volume no", "volume number"],
    "property.page_no": ["page no", "page numbers", "pages"],
    "property.survey_no": ["survey no", "survey number"],
    "property.landmark": ["land mark", "landmark"],
    "property.latitude_north": ["latitude(north)", "latitude north", "lat north"],
    "property.latitude_east": ["latitude(east)", "latitude east", "lat east"],
    "property.taluka": ["taluka", "tahsil", "tehsil"],

    # Property Boundaries
    "property.boundaries.north": ["north", "boundary north", "bounded on north"],
    "property.boundaries.south": ["south", "boundary south", "bounded on south"],
    "property.boundaries.east": ["east", "boundary east", "bounded on east"],
    "property.boundaries.west": ["west", "boundary west", "bounded on west"],

    # Property Valuation / Searching
    "property.valuation_independent": ["market value of the property as per independent enquiry:"],
    "property.valuation_empanelled": ["market value of the property as per empanelled value :", "valuation of the property"],
    "property.valuation_date": ["date of valuation", "valuation date"],
    "property.valuer_name": ["valuation done by", "valuer name", "surveyor name"],
    "property.search_report_date": ["date of searching report", "searching report date"],
    "property.search_by": ["searching made by", "advocate search"],
    "property.security_interest_holder": ["security interest holder name"],
    "property.bsr_code": ["bsr code"],
    "property.security_interest_creation_date": ["security interest creation date"],
    "property.document_type": ["doccument type", "document type"],
    "property.document_registration_date": ["doccument/registration date", "registration date"],

    # Project Cost
    "property.project_cost.construction_cost": ["construction", "construction cost", "construction cost rs."],
    "property.project_cost.renovation_cost": ["renovation", "renovation cost"],
    "property.project_cost.repairs_cost": ["repairs", "repairs cost"],
    "property.project_cost.furnishing_cost": ["purchase extension furnishing", "furnishing cost"],

    # Assets of Borrower
    "assets.bank_deposit": ["bank & other deposit", "bank deposits", "savings"],
    "assets.life_insurance": ["life insurance policies", "lic policies", "insurance"],
    "assets.other_movable": ["other movable property", "movable property"],
    "assets.immovable_house": ["house", "house property"],
    "assets.immovable_land": ["land", "land property"],

    # Liabilities of Borrower
    "liabilities.from_employer": ["from employer", "employer loan"],
    "liabilities.from_family_friends": ["from family & friends", "family loan"],
    "liabilities.from_provident_fund": ["from provident fund", "pf loan"],
    "liabilities.from_other_source": ["from other source", "other liabilities"],
}


# ==============================================================================
# CENTRAL NORMALIZED DATA STRUCTURE
# ==============================================================================

class NormalizedLoanData:
    """
    Central normalized container representing all extracted loan data.
    Organized into 10 explicit categories matching the Excel workbook.
    """

    def __init__(self):
        # 1. Borrower Master
        self.borrower: Dict[str, Any] = {
            "full_name": NormalizedField("borrower.full_name"),
            "title": NormalizedField("borrower.title"),
            "first_name": NormalizedField("borrower.first_name"),
            "surname": NormalizedField("borrower.surname"),
            "dob": NormalizedField("borrower.dob"),
            "gender": NormalizedField("borrower.gender"),
            "father_husband_name": NormalizedField("borrower.father_husband_name"),
            "qualification": NormalizedField("borrower.qualification"),
            "marital_status": NormalizedField("borrower.marital_status"),
            "num_dependents": NormalizedField("borrower.num_dependents"),
            "mobile_no": NormalizedField("borrower.mobile_no"),
            "pan_no": NormalizedField("borrower.pan_no"),
            "aadhar_no": NormalizedField("borrower.aadhar_no"),
            "voter_card_no": NormalizedField("borrower.voter_card_no"),
            "driving_license_no": NormalizedField("borrower.driving_license_no"),
            "resident_address": {
                "vill": NormalizedField("borrower.resident_address.vill"),
                "post": NormalizedField("borrower.resident_address.post"),
                "dist": NormalizedField("borrower.resident_address.dist"),
                "pin": NormalizedField("borrower.resident_address.pin"),
            },
            "present_posting": {
                "org_name": NormalizedField("borrower.present_posting.org_name"),
                "vill": NormalizedField("borrower.present_posting.vill"),
                "post": NormalizedField("borrower.present_posting.post"),
                "dist": NormalizedField("borrower.present_posting.dist"),
                "pin": NormalizedField("borrower.present_posting.pin"),
            },
            "employer": {
                "employer_name": NormalizedField("borrower.employer.employer_name"),
                "vill": NormalizedField("borrower.employer.vill"),
                "post": NormalizedField("borrower.employer.post"),
                "dist": NormalizedField("borrower.employer.dist"),
                "pin": NormalizedField("borrower.employer.pin"),
            },
            "service": {
                "length_of_service": NormalizedField("borrower.service.length_of_service"),
                "designation": NormalizedField("borrower.service.designation"),
                "date_of_joining": NormalizedField("borrower.service.date_of_joining"),
                "date_of_retirement": NormalizedField("borrower.service.date_of_retirement"),
            }
        }

        # 2 & 3. Guarantors (Strictly Segregated Guarantor 1 & Guarantor 2)
        self.guarantors: List[Dict[str, Any]] = [
            self._create_empty_guarantor(1),
            self._create_empty_guarantor(2)
        ]

        # 4. Loan Request & Terms
        self.loan: Dict[str, Any] = {
            "branch": NormalizedField("loan.branch"),
            "application_date": NormalizedField("loan.application_date"),
            "application_day": NormalizedField("loan.application_day"),
            "application_month": NormalizedField("loan.application_month"),
            "application_year": NormalizedField("loan.application_year"),
            "mortgage_branch": NormalizedField("loan.mortgage_branch"),
            "salary_ac_no": NormalizedField("loan.salary_ac_no"),
            "loan_account_no": NormalizedField("loan.loan_account_no"),
            "loan_amount_requested": NormalizedField("loan.loan_amount_requested"),
            "term_months": NormalizedField("loan.term_months"),
            "rate_of_interest": NormalizedField("loan.rate_of_interest"),
            "emi": NormalizedField("loan.emi"),
            "other_emi": NormalizedField("loan.other_emi"),
            "moratorium_period": NormalizedField("loan.moratorium_period"),
            "emi_starting_from": NormalizedField("loan.emi_starting_from"),
            "purpose": NormalizedField("loan.purpose"),
        }

        # 5. Property / Land Master
        self.property: Dict[str, Any] = {
            "inspection_date": NormalizedField("property.inspection_date"),
            "mortgage_branch": NormalizedField("property.mortgage_branch"),
            "sub_registry_office": NormalizedField("property.sub_registry_office"),
            "plan_approved_by": NormalizedField("property.plan_approved_by"),
            "flat_no": NormalizedField("property.flat_no"),
            "house_no": NormalizedField("property.house_no"),
            "premises_no": NormalizedField("property.premises_no"),
            "building_name": NormalizedField("property.building_name"),
            "plot_dag_no": NormalizedField("property.plot_dag_no"),
            "khatian_no": NormalizedField("property.khatian_no"),
            "jl_no": NormalizedField("property.jl_no"),
            "mouza": NormalizedField("property.mouza"),
            "block": NormalizedField("property.block"),
            "ps": NormalizedField("property.ps"),
            "post": NormalizedField("property.post"),
            "pin": NormalizedField("property.pin"),
            "dist": NormalizedField("property.dist"),
            "area": NormalizedField("property.area"),
            "classification": NormalizedField("property.classification"),
            "proposed_buildup_area": NormalizedField("property.proposed_buildup_area"),
            "purchase_price": NormalizedField("property.purchase_price"),
            "purchase_date": NormalizedField("property.purchase_date"),
            "municipality_gp": NormalizedField("property.municipality_gp"),
            "street_name": NormalizedField("property.street_name"),
            "deed_no": NormalizedField("property.deed_no"),
            "book_no": NormalizedField("property.book_no"),
            "vol_no": NormalizedField("property.vol_no"),
            "page_no": NormalizedField("property.page_no"),
            "survey_no": NormalizedField("property.survey_no"),
            "landmark": NormalizedField("property.landmark"),
            "latitude_north": NormalizedField("property.latitude_north"),
            "latitude_east": NormalizedField("property.latitude_east"),
            "taluka": NormalizedField("property.taluka"),
            "boundaries": {
                "north": NormalizedField("property.boundaries.north"),
                "south": NormalizedField("property.boundaries.south"),
                "east": NormalizedField("property.boundaries.east"),
                "west": NormalizedField("property.boundaries.west"),
            },
            "valuation_independent": NormalizedField("property.valuation_independent"),
            "valuation_empanelled": NormalizedField("property.valuation_empanelled"),
            "valuation_date": NormalizedField("property.valuation_date"),
            "valuer_name": NormalizedField("property.valuer_name"),
            "search_report_date": NormalizedField("property.search_report_date"),
            "search_by": NormalizedField("property.search_by"),
            "security_interest_holder": NormalizedField("property.security_interest_holder", value="BANGIYA GRAMIN VIKASH BANK"),
            "bsr_code": NormalizedField("property.bsr_code"),
            "security_interest_creation_date": NormalizedField("property.security_interest_creation_date"),
            "document_type": NormalizedField("property.document_type"),
            "document_registration_date": NormalizedField("property.document_registration_date"),
            "project_cost": {
                "construction_cost": NormalizedField("property.project_cost.construction_cost"),
                "construction_incidental": NormalizedField("property.project_cost.construction_incidental"),
                "construction_contingency": NormalizedField("property.project_cost.construction_contingency"),
                "renovation_cost": NormalizedField("property.project_cost.renovation_cost"),
                "repairs_cost": NormalizedField("property.project_cost.repairs_cost"),
                "furnishing_cost": NormalizedField("property.project_cost.furnishing_cost"),
            }
        }

        # 6. Salary & Income
        self.salary: Dict[str, Any] = {
            "salary_month": NormalizedField("salary.salary_month"),
            "gross_salary": NormalizedField("salary.gross_salary"),
            "basic_pay": NormalizedField("salary.basic_pay"),
            "da": NormalizedField("salary.da"),
            "hra": NormalizedField("salary.hra"),
            "others_allowance": NormalizedField("salary.others_allowance"),
            "deductions": {
                "pf": NormalizedField("salary.deduction_pf"),
                "it": NormalizedField("salary.deduction_it"),
                "others": NormalizedField("salary.deduction_others"),
            },
            "family_expenditure": NormalizedField("salary.family_expenditure"),
            "p_tax_last_paid": NormalizedField("salary.p_tax_last_paid"),
            "form16_last_submitted": NormalizedField("salary.form16_last_submitted"),
        }

        # 7. Assets and Liabilities (Borrower)
        self.assets: Dict[str, Any] = {
            "bank_deposit": NormalizedField("assets.bank_deposit"),
            "life_insurance": NormalizedField("assets.life_insurance"),
            "other_movable": NormalizedField("assets.other_movable"),
            "immovable_house": NormalizedField("assets.immovable_house"),
            "immovable_land": NormalizedField("assets.immovable_land"),
        }
        self.liabilities: Dict[str, Any] = {
            "from_employer": NormalizedField("liabilities.from_employer"),
            "from_family_friends": NormalizedField("liabilities.from_family_friends"),
            "from_provident_fund": NormalizedField("liabilities.from_provident_fund"),
            "from_other_source": NormalizedField("liabilities.from_other_source"),
        }

        # 8. Dependents (Up to 3 family members)
        self.dependents: List[Dict[str, NormalizedField]] = [
            {"name": NormalizedField(f"dependents[{i}].name"),
             "age": NormalizedField(f"dependents[{i}].age"),
             "relationship": NormalizedField(f"dependents[{i}].relationship")}
            for i in range(3)
        ]

        # 9. References (2 personal references)
        self.references: List[Dict[str, NormalizedField]] = [
            {
                "name": NormalizedField(f"references[{i}].name"),
                "vill": NormalizedField(f"references[{i}].vill"),
                "post": NormalizedField(f"references[{i}].post"),
                "dist": NormalizedField(f"references[{i}].dist"),
                "pin": NormalizedField(f"references[{i}].pin"),
                "mobile_no": NormalizedField(f"references[{i}].mobile_no"),
            }
            for i in range(2)
        ]

        # 10. Audit Metadata & Other Unmapped Information
        self.other_info: Dict[str, Any] = {
            "source_documents": [],
            "processed_count": 0,
            "engine_used": None,
            "created_at": datetime.now().isoformat(),
            "unmapped_fields": []
        }

    def _create_empty_guarantor(self, index: int) -> Dict[str, Any]:
        """Creates an isolated Guarantor container."""
        prefix = f"guarantor_{index}"
        return {
            "index": index,
            "name": NormalizedField(f"{prefix}.name"),
            "father_husband_name": NormalizedField(f"{prefix}.father_husband_name"),
            "dob": NormalizedField(f"{prefix}.dob"),
            "occupation": NormalizedField(f"{prefix}.occupation"),
            "monthly_income": NormalizedField(f"{prefix}.monthly_income"),
            "p_tax_paid_upto": NormalizedField(f"{prefix}.p_tax_paid_upto"),
            "mobile_no": NormalizedField(f"{prefix}.mobile_no"),
            "qualification": NormalizedField(f"{prefix}.qualification"),
            "account_no": NormalizedField(f"{prefix}.account_no"),
            "resident_address": {
                "vill": NormalizedField(f"{prefix}.resident_address.vill"),
                "post": NormalizedField(f"{prefix}.resident_address.post"),
                "dist": NormalizedField(f"{prefix}.resident_address.dist"),
                "pin": NormalizedField(f"{prefix}.resident_address.pin"),
            },
            "work_address": {
                "vill": NormalizedField(f"{prefix}.work_address.vill"),
                "post": NormalizedField(f"{prefix}.work_address.post"),
                "dist": NormalizedField(f"{prefix}.work_address.dist"),
                "pin": NormalizedField(f"{prefix}.work_address.pin"),
            },
            "assets": [
                {"description": NormalizedField(f"{prefix}.asset[{i}].desc"),
                 "amount": NormalizedField(f"{prefix}.asset[{i}].amount")}
                for i in range(4)
            ],
            "liabilities": [
                {"description": NormalizedField(f"{prefix}.liability[{i}].desc"),
                 "amount": NormalizedField(f"{prefix}.liability[{i}].amount")}
                for i in range(4)
            ],
            "dependents": [
                {"name": NormalizedField(f"{prefix}.dependent[{i}].name"),
                 "age": NormalizedField(f"{prefix}.dependent[{i}].age"),
                 "relationship": NormalizedField(f"{prefix}.dependent[{i}].relationship")}
                for i in range(3)
            ]
        }

    def set_by_path(
        self,
        path: str,
        value: Any,
        source_doc: Optional[str] = None,
        confidence: Optional[float] = None,
        evidence: Optional[str] = None,
        raw_value: Optional[Any] = None
    ) -> bool:
        """Dynamically navigates and updates a field by dot-separated path."""
        parts = path.split(".")
        target: Any = self
        for part in parts[:-1]:
            if hasattr(target, part):
                target = getattr(target, part)
            elif isinstance(target, dict) and part in target:
                target = target[part]
            elif isinstance(target, list) and part.isdigit():
                idx = int(part)
                if 0 <= idx < len(target):
                    target = target[idx]
                else:
                    return False
            else:
                return False

        last_part = parts[-1]
        field_obj = None
        if hasattr(target, last_part):
            field_obj = getattr(target, last_part)
        elif isinstance(target, dict):
            if last_part in target:
                field_obj = target[last_part]
            elif last_part == "full_name" and "name" in target:
                field_obj = target["name"]
            elif last_part == "name" and "full_name" in target:
                field_obj = target["full_name"]

        if isinstance(field_obj, NormalizedField):
            field_obj.update(
                new_value=value,
                source_document=source_doc,
                confidence=confidence,
                evidence=evidence,
                raw_value=raw_value
            )
            return True
        return False

    def to_dict(self) -> Dict[str, Any]:
        """Converts entire data structure into a nested dictionary with metadata."""
        def serialize(obj):
            if isinstance(obj, NormalizedField):
                return obj.to_dict()
            elif isinstance(obj, dict):
                return {k: serialize(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [serialize(item) for item in obj]
            return obj

        return {
            "borrower": serialize(self.borrower),
            "guarantors": serialize(self.guarantors),
            "loan": serialize(self.loan),
            "property": serialize(self.property),
            "salary": serialize(self.salary),
            "assets": serialize(self.assets),
            "liabilities": serialize(self.liabilities),
            "dependents": serialize(self.dependents),
            "references": serialize(self.references),
            "other_info": serialize(self.other_info)
        }

    def to_flat_values(self) -> Dict[str, Any]:
        """Returns clean flat dictionary of {path: value} for immediate use/viewing."""
        flat = {}
        def recurse(prefix, obj):
            if isinstance(obj, NormalizedField):
                flat[prefix] = obj.value
            elif isinstance(obj, dict):
                for k, v in obj.items():
                    p = f"{prefix}.{k}" if prefix else k
                    recurse(p, v)
            elif isinstance(obj, list):
                for idx, item in enumerate(obj):
                    recurse(f"{prefix}[{idx}]", item)

        recurse("borrower", self.borrower)
        recurse("guarantors", self.guarantors)
        recurse("loan", self.loan)
        recurse("property", self.property)
        recurse("salary", self.salary)
        recurse("assets", self.assets)
        recurse("liabilities", self.liabilities)
        recurse("dependents", self.dependents)
        recurse("references", self.references)
        return flat

    def to_excel_cell_map(self) -> Dict[str, Dict[str, Any]]:
        """
        Maps normalized values directly to target Excel coordinates identified
        in the Excel Mapping Specification.
        Returns format: { 'SHEET_NAME': { 'CELL': value } }
        """
        cell_map = {
            "BORROWERDETAILS": {},
            "GURANTORSDETAILS": {},
            "LANDDETAILS": {}
        }

        # --- BORROWERDETAILS ---
        bd = cell_map["BORROWERDETAILS"]
        bd["K7"] = self.loan["branch"].value
        bd["K8"] = self.loan["application_date"].value
        bd["K9"] = self.loan["application_date"].value
        bd["K10"] = self.loan["application_day"].value
        bd["K11"] = self.loan["application_month"].value
        bd["K12"] = self.loan["application_year"].value
        bd["K13"] = self.loan["mortgage_branch"].value
        bd["K14"] = self.loan["salary_ac_no"].value
        bd["K15"] = self.loan["loan_account_no"].value
        bd["K16"] = self.loan["loan_amount_requested"].value
        bd["K17"] = self.loan["term_months"].value
        bd["K18"] = self.loan["rate_of_interest"].value
        bd["K19"] = self.loan["emi"].value
        bd["K20"] = self.loan["other_emi"].value
        bd["K21"] = self.loan["moratorium_period"].value
        bd["K22"] = self.loan["emi_starting_from"].value
        bd["K23"] = self.loan["purpose"].value

        bd["K24"] = self.borrower["full_name"].value
        bd["K25"] = self.borrower["title"].value
        bd["K26"] = self.borrower["first_name"].value
        bd["K27"] = self.borrower["surname"].value
        bd["K28"] = self.borrower["dob"].value
        bd["K29"] = self.borrower["gender"].value
        bd["K30"] = self.borrower["father_husband_name"].value
        bd["K31"] = self.borrower["qualification"].value
        bd["K32"] = self.borrower["marital_status"].value
        bd["K33"] = self.borrower["num_dependents"].value

        # Resident Address
        bd["K35"] = self.borrower["resident_address"]["vill"].value
        bd["K36"] = self.borrower["resident_address"]["post"].value
        bd["K37"] = self.borrower["resident_address"]["dist"].value
        bd["K38"] = self.borrower["resident_address"]["pin"].value
        bd["K39"] = self.borrower["mobile_no"].value

        # Posting Address
        bd["K41"] = self.borrower["present_posting"]["org_name"].value
        bd["K42"] = self.borrower["present_posting"]["vill"].value
        bd["K43"] = self.borrower["present_posting"]["post"].value
        bd["K44"] = self.borrower["present_posting"]["dist"].value
        bd["K45"] = self.borrower["present_posting"]["pin"].value

        # Employer Address
        bd["K47"] = self.borrower["employer"]["employer_name"].value
        bd["K48"] = self.borrower["employer"]["vill"].value
        bd["K49"] = self.borrower["employer"]["post"].value
        bd["K50"] = self.borrower["employer"]["dist"].value
        bd["K51"] = self.borrower["employer"]["pin"].value

        # Service Details
        bd["K52"] = self.borrower["service"]["length_of_service"].value
        bd["K53"] = self.borrower["service"]["designation"].value
        bd["K54"] = self.borrower["service"]["date_of_joining"].value
        bd["K55"] = self.borrower["service"]["date_of_retirement"].value

        # References
        if len(self.references) > 0:
            bd["K57"] = self.references[0]["name"].value
            bd["K58"] = self.references[0]["vill"].value
            bd["K59"] = self.references[0]["post"].value
            bd["K60"] = self.references[0]["dist"].value
            bd["K61"] = self.references[0]["pin"].value
            bd["K62"] = self.references[0]["mobile_no"].value
        if len(self.references) > 1:
            bd["K63"] = self.references[1]["name"].value
            bd["K64"] = self.references[1]["vill"].value
            bd["K65"] = self.references[1]["post"].value
            bd["K66"] = self.references[1]["dist"].value
            bd["K67"] = self.references[1]["pin"].value
            bd["K68"] = self.references[1]["mobile_no"].value

        # KYC
        bd["K70"] = self.borrower["pan_no"].value
        bd["K71"] = self.borrower["voter_card_no"].value
        bd["K72"] = self.borrower["aadhar_no"].value

        # Salary
        bd["K74"] = self.salary["salary_month"].value
        bd["K75"] = self.salary["gross_salary"].value
        bd["K76"] = self.salary["basic_pay"].value
        bd["K77"] = self.salary["da"].value
        bd["K78"] = self.salary["hra"].value
        bd["K79"] = self.salary["others_allowance"].value
        # Note: K80 is formula =K76+K77+K78+K79
        bd["K82"] = self.salary["deductions"]["pf"].value
        bd["K83"] = self.salary["deductions"]["it"].value
        bd["K84"] = self.salary["deductions"]["others"].value
        # Note: K85 is formula =K82+K83+K84
        # Note: K86 is formula =K80-K85
        bd["K87"] = self.salary["family_expenditure"].value
        bd["K88"] = self.salary["p_tax_last_paid"].value

        # Assets
        bd["K90"] = self.assets["bank_deposit"].value
        bd["K91"] = self.assets["life_insurance"].value
        bd["K92"] = self.assets["other_movable"].value
        bd["K96"] = self.assets["immovable_house"].value
        bd["K97"] = self.assets["immovable_land"].value
        # Note: K99 is formula

        # Liabilities
        bd["K101"] = self.liabilities["from_employer"].value
        bd["K102"] = self.liabilities["from_family_friends"].value
        bd["K103"] = self.liabilities["from_provident_fund"].value
        bd["K104"] = self.liabilities["from_other_source"].value
        # Note: K107 is formula, K108 is formula
        bd["K109"] = self.salary["form16_last_submitted"].value

        # Dependents
        if len(self.dependents) > 0:
            bd["K111"] = self.dependents[0]["name"].value
            bd["K112"] = self.dependents[0]["age"].value
            bd["K113"] = self.dependents[0]["relationship"].value
        if len(self.dependents) > 1:
            bd["K114"] = self.dependents[1]["name"].value
            bd["K115"] = self.dependents[1]["age"].value
            bd["K116"] = self.dependents[1]["relationship"].value
        if len(self.dependents) > 2:
            bd["K117"] = self.dependents[2]["name"].value
            bd["K118"] = self.dependents[2]["age"].value
            bd["K119"] = self.dependents[2]["relationship"].value

        # --- GURANTORSDETAILS ---
        gd = cell_map["GURANTORSDETAILS"]
        # Guarantor 1 (Rows 1..45)
        g1 = self.guarantors[0]
        gd["K2"] = g1["name"].value
        gd["K3"] = g1["father_husband_name"].value
        gd["K4"] = g1["dob"].value
        gd["K5"] = g1["occupation"].value
        gd["K6"] = g1["monthly_income"].value
        gd["K8"] = g1["resident_address"]["vill"].value
        gd["K9"] = g1["resident_address"]["post"].value
        gd["K10"] = g1["resident_address"]["dist"].value
        gd["K11"] = g1["resident_address"]["pin"].value
        gd["K12"] = g1["p_tax_paid_upto"].value
        gd["K13"] = g1["mobile_no"].value
        gd["K15"] = g1["work_address"]["vill"].value
        gd["K16"] = g1["work_address"]["post"].value
        gd["K17"] = g1["work_address"]["dist"].value
        gd["K18"] = g1["work_address"]["pin"].value
        gd["K19"] = g1["qualification"].value
        gd["K20"] = g1["account_no"].value

        # Guarantor 2 (Rows 46..90)
        g2 = self.guarantors[1]
        gd["K47"] = g2["name"].value
        gd["K48"] = g2["father_husband_name"].value
        gd["K49"] = g2["dob"].value
        gd["K50"] = g2["occupation"].value
        gd["K51"] = g2["monthly_income"].value
        gd["K53"] = g2["resident_address"]["vill"].value
        gd["K54"] = g2["resident_address"]["post"].value
        gd["K55"] = g2["resident_address"]["dist"].value
        gd["K56"] = g2["resident_address"]["pin"].value
        gd["K57"] = g2["p_tax_paid_upto"].value
        gd["K58"] = g2["mobile_no"].value
        gd["K60"] = g2["work_address"]["vill"].value
        gd["K61"] = g2["work_address"]["post"].value
        gd["K62"] = g2["work_address"]["dist"].value
        gd["K63"] = g2["work_address"]["pin"].value
        gd["K64"] = g2["qualification"].value
        gd["K65"] = g2["account_no"].value

        # --- LANDDETAILS ---
        ld = cell_map["LANDDETAILS"]
        ld["K2"] = self.property["inspection_date"].value
        ld["K3"] = self.property["mortgage_branch"].value
        ld["K4"] = self.property["sub_registry_office"].value
        ld["K5"] = self.property["plan_approved_by"].value
        ld["K6"] = self.property["flat_no"].value
        ld["K7"] = self.property["house_no"].value
        ld["K8"] = self.property["premises_no"].value
        ld["K9"] = self.property["building_name"].value
        ld["K10"] = self.property["plot_dag_no"].value
        ld["K11"] = self.property["khatian_no"].value
        ld["K12"] = self.property["jl_no"].value
        ld["K13"] = self.property["mouza"].value
        ld["K14"] = self.property["block"].value
        ld["K15"] = self.property["ps"].value
        ld["K16"] = self.property["post"].value
        ld["K17"] = self.property["pin"].value
        ld["K18"] = self.property["dist"].value
        ld["K19"] = self.property["area"].value
        ld["K20"] = self.property["classification"].value
        ld["K21"] = self.property["proposed_buildup_area"].value
        ld["K22"] = self.property["purchase_price"].value
        ld["K23"] = self.property["purchase_date"].value
        ld["K24"] = self.property["municipality_gp"].value
        ld["K25"] = self.property["street_name"].value
        ld["K26"] = self.property["deed_no"].value
        ld["K27"] = self.property["book_no"].value
        ld["K28"] = self.property["vol_no"].value
        ld["K29"] = self.property["page_no"].value
        ld["K31"] = self.property["survey_no"].value
        ld["K32"] = self.property["landmark"].value
        ld["K33"] = self.property["latitude_north"].value
        ld["K34"] = self.property["latitude_east"].value
        ld["K35"] = self.property["taluka"].value

        # Boundaries
        ld["K42"] = self.property["boundaries"]["north"].value
        ld["K43"] = self.property["boundaries"]["south"].value
        ld["K44"] = self.property["boundaries"]["east"].value
        ld["K45"] = self.property["boundaries"]["west"].value

        # Valuations & Legal
        ld["K53"] = self.property["valuation_independent"].value
        ld["K54"] = self.property["valuation_empanelled"].value
        ld["K80"] = self.property["search_by"].value
        ld["K81"] = self.property["search_report_date"].value
        ld["K84"] = self.property["valuation_date"].value
        ld["K85"] = self.property["valuer_name"].value

        # Security Interest
        ld["K96"] = self.property["security_interest_holder"].value
        ld["K97"] = self.property["bsr_code"].value
        ld["K110"] = self.property["security_interest_creation_date"].value
        ld["K111"] = self.property["document_type"].value
        ld["K112"] = self.property["deed_no"].value
        ld["K113"] = self.property["document_registration_date"].value

        # Project Cost Grid (K, M, O cols)
        ld["K120"] = self.property["project_cost"]["construction_cost"].value
        ld["M120"] = self.property["project_cost"]["construction_incidental"].value
        ld["O120"] = self.property["project_cost"]["construction_contingency"].value
        ld["K121"] = self.property["project_cost"]["renovation_cost"].value
        ld["K122"] = self.property["project_cost"]["repairs_cost"].value
        ld["K123"] = self.property["project_cost"]["furnishing_cost"].value

        # Filter out keys with None so we don't unnecessarily overwrite existing template formulas/cells
        return {
            sheet: {cell: val for cell, val in cells.items() if val is not None}
            for sheet, cells in cell_map.items()
        }


# ==============================================================================
# NORMALIZATION ENGINE
# ==============================================================================

class NormalizationEngine:
    """
    Ingests output from the existing OCR/LLM extraction pipeline,
    infers entity associations (Borrower vs Guarantor 1 vs Guarantor 2 vs Land),
    standardizes variations into canonical fields, validates values,
    and returns a fully populated NormalizedLoanData object.
    """

    def __init__(self):
        # Inverted index: {alias_lower: canonical_path}
        self.alias_index: Dict[str, str] = {}
        for canonical, aliases in LABEL_ALIASES.items():
            for alias in aliases:
                self.alias_index[alias.strip().lower()] = canonical

    def infer_entity_role(self, doc_name: str, doc_type: str, text: str = "") -> str:
        """
        Determines target entity (borrower, guarantor_1, guarantor_2, property, salary, general)
        based on file name, document classification, and content markers.
        """
        combined = f"{doc_name} {doc_type} {text[:300]}".lower()

        # Guarantor check
        if any(k in combined for k in ["guarantor 2", "guarantor_2", "g2", "second guarantor"]):
            return "guarantor_2"
        elif any(k in combined for k in ["guarantor 1", "guarantor_1", "guarantor", "surety", "g1"]):
            return "guarantor_1"

        # Property / Land check
        if any(k in combined for k in ["deed", "sale deed", "land", "plot", "khatian", "mouza", "valuation", "search report"]):
            return "property"

        # Salary / Income check
        if any(k in combined for k in ["salary", "payslip", "pay slip", "form 16", "basic pay", "earnings"]):
            return "salary"

        # Default fallback for KYC / Identity / Application is Borrower
        return "borrower"

    def normalize(
        self,
        extraction_input: Union[Dict[str, Any], List[Dict[str, Any]]],
        document_role_overrides: Optional[Dict[str, str]] = None
    ) -> NormalizedLoanData:
        """
        Main entry point for normalization.
        Accepts:
          - API response dict: {"documents": [...], "processed_files": [...]}
          - ai_extractor result dict: {"documents": [...]}
          - List of document objects directly: [{...}, {...}]
        Optionally accepts `document_role_overrides` mapping: {"filename": "guarantor_1"}
        """
        role_overrides = document_role_overrides or {}
        normalized = NormalizedLoanData()

        # Parse documents list from input
        documents = []
        if isinstance(extraction_input, dict):
            documents = extraction_input.get("documents", [])
            normalized.other_info["engine_used"] = extraction_input.get("engine_used")
        elif isinstance(extraction_input, list):
            documents = extraction_input

        normalized.other_info["processed_count"] = len(documents)

        # Iterate over documents in batch
        for doc in documents:
            doc_name = doc.get("document_name", "Document")
            doc_type = doc.get("document_type", "Document")
            fields = doc.get("fields", {})
            ocr_text = doc.get("ocr_text", "")
            confidence = doc.get("confidence", 0.90)

            normalized.other_info["source_documents"].append({
                "name": doc_name,
                "type": doc_type,
                "field_count": len(fields)
            })

            # Determine entity role for this document
            entity_role = role_overrides.get(doc_name) or self.infer_entity_role(doc_name, doc_type, ocr_text)

            # Process fields inside document
            if isinstance(fields, dict):
                items = fields.items()
            elif isinstance(fields, list):
                # Handle list of field objects or dicts
                items = []
                for item in fields:
                    if isinstance(item, dict):
                        items.extend(item.items())
            else:
                items = []

            for raw_label, raw_val in items:
                if raw_val is None or str(raw_val).strip() == "":
                    continue

                clean_label = str(raw_label).strip().lower()
                canonical_path = self.alias_index.get(clean_label)

                evidence_snippet = self._find_evidence_snippet(ocr_text, str(raw_val))

                if canonical_path:
                    # Reroute to Guarantor 1 or Guarantor 2 if this document is flagged as a guarantor
                    final_path = canonical_path
                    if entity_role in ["guarantor_1", "guarantor"]:
                        if canonical_path.startswith("borrower."):
                            sub_path = canonical_path[len("borrower."):]
                            final_path = f"guarantors.0.{sub_path}"
                        elif canonical_path.startswith("guarantor."):
                            sub_path = canonical_path[len("guarantor."):]
                            final_path = f"guarantors.0.{sub_path}"
                    elif entity_role == "guarantor_2":
                        if canonical_path.startswith("borrower."):
                            sub_path = canonical_path[len("borrower."):]
                            final_path = f"guarantors.1.{sub_path}"
                        elif canonical_path.startswith("guarantor."):
                            sub_path = canonical_path[len("guarantor."):]
                            final_path = f"guarantors.1.{sub_path}"

                    normalized.set_by_path(
                        final_path,
                        value=raw_val,
                        source_doc=doc_name,
                        confidence=confidence,
                        evidence=evidence_snippet,
                        raw_value=raw_val
                    )
                else:
                    # Store as unmapped field to guarantee zero data loss
                    normalized.other_info["unmapped_fields"].append({
                        "label": raw_label,
                        "value": raw_val,
                        "source_document": doc_name,
                        "entity_role": entity_role,
                        "evidence": evidence_snippet
                    })

        # Post-Processing: Derive Sub-Fields
        self._post_process_derivatives(normalized)

        # Validate normalized data
        self._validate_normalized_data(normalized)

        return normalized

    def _find_evidence_snippet(self, ocr_text: str, val_str: str, max_chars: int = 120) -> Optional[str]:
        """Locates the line in raw OCR text that contains the extracted value."""
        if not ocr_text or not val_str or len(val_str) < 3:
            return None
        clean_val = val_str.strip().lower()
        for line in ocr_text.splitlines():
            if clean_val in line.lower():
                return line.strip()[:max_chars]
        return None

    def _post_process_derivatives(self, data: NormalizedLoanData):
        """
        Derives split sub-fields required by downstream templates
        (e.g., Name Breakdown -> Title, First Name, Surname for CERSAI;
         Date Breakdown -> Day, Month, Year for AGRREMENTUNDERBHBL).
        """
        # 1. Borrower Name Decomposition (K24 -> K25, K26, K27)
        full_name = data.borrower["full_name"].value
        if full_name and not data.borrower["first_name"].value:
            parts = str(full_name).strip().split()
            title = ""
            first = ""
            sur = ""
            if parts and parts[0].upper() in ["MR", "MR.", "MRS", "MRS.", "MS", "MS.", "SHRI", "SMT", "MD", "DR", "DR."]:
                title = parts[0]
                parts = parts[1:]
            if len(parts) == 1:
                first = parts[0]
            elif len(parts) >= 2:
                first = " ".join(parts[:-1])
                sur = parts[-1]

            if title: data.borrower["title"].update(title, source_document="Derived from Full Name", confidence=1.0)
            if first: data.borrower["first_name"].update(first, source_document="Derived from Full Name", confidence=1.0)
            if sur: data.borrower["surname"].update(sur, source_document="Derived from Full Name", confidence=1.0)

        # 2. Date Decomposition (K9 -> K10, K11, K12)
        app_date = data.loan["application_date"].value
        if app_date and not data.loan["application_day"].value:
            d_obj = self._parse_date(str(app_date))
            if d_obj:
                day_num = d_obj.day
                ordinal = self._ordinal_suffix(day_num)
                month_name = d_obj.strftime("%B").lower()
                year_num = d_obj.year

                data.loan["application_day"].update(ordinal, source_document="Derived from Date", confidence=1.0)
                data.loan["application_month"].update(month_name, source_document="Derived from Date", confidence=1.0)
                data.loan["application_year"].update(year_num, source_document="Derived from Date", confidence=1.0)

        # 3. Address Decomposition: if address string was extracted in one line
        # e.g. "Vill - Kalipur, PO - Kalipur, Dist - Burdwan, PIN - 713101"
        res_addr = data.borrower["resident_address"]
        if not res_addr["vill"].value:
            pass # can extract components using regex if needed

    def _parse_date(self, text: str) -> Optional[datetime]:
        """Attempts to parse common date formats."""
        for fmt in ["%d.%m.%Y", "%d/%m/%Y", "%d-%m-%Y", "%Y-%m-%d", "%d %B %Y", "%d %b %Y"]:
            try:
                return datetime.strptime(text.strip(), fmt)
            except ValueError:
                continue
        # Try regex extract
        m = re.search(r'(\d{1,2})[\/\.\-](\d{1,2})[\/\.\-](\d{4})', text)
        if m:
            try:
                return datetime(int(m.group(3)), int(m.group(2)), int(m.group(1)))
            except ValueError:
                pass
        return None

    def _ordinal_suffix(self, day: int) -> str:
        """Returns ordinal string e.g. 1 -> '1st', 2 -> '2nd', 3 -> '3rd', 4 -> '4th'."""
        if 11 <= (day % 100) <= 13:
            suffix = 'th'
        else:
            suffix = {1: 'st', 2: 'nd', 3: 'rd'}.get(day % 10, 'th')
        return f"{day}{suffix}"

    def _validate_normalized_data(self, data: NormalizedLoanData):
        """Validates canonical formats: PAN, Aadhaar, Phone, Pin, etc."""
        # Validate Borrower PAN
        pan = data.borrower["pan_no"].value
        if pan:
            pan_clean = str(pan).strip().upper()
            if not re.match(r'^[A-Z]{5}[0-9]{4}[A-Z]$', pan_clean):
                data.borrower["pan_no"].is_valid = False
                data.borrower["pan_no"].validation_note = "Invalid PAN pattern (expected 5 letters, 4 digits, 1 letter)"
            else:
                data.borrower["pan_no"].value = pan_clean

        # Validate Borrower Aadhaar
        aadhaar = data.borrower["aadhar_no"].value
        if aadhaar:
            adh_digits = re.sub(r'\D', '', str(aadhaar))
            if len(adh_digits) != 12:
                data.borrower["aadhar_no"].is_valid = False
                data.borrower["aadhar_no"].validation_note = "Aadhaar must contain exactly 12 digits"
            else:
                data.borrower["aadhar_no"].value = f"{adh_digits[:4]} {adh_digits[4:8]} {adh_digits[8:]}"

        # Validate Borrower Mobile
        mob = data.borrower["mobile_no"].value
        if mob:
            mob_digits = re.sub(r'\D', '', str(mob))
            if len(mob_digits) > 10 and mob_digits.startswith("91"):
                mob_digits = mob_digits[2:]
            if len(mob_digits) != 10:
                data.borrower["mobile_no"].is_valid = False
                data.borrower["mobile_no"].validation_note = "Mobile number should be 10 digits"
            else:
                data.borrower["mobile_no"].value = mob_digits

        # Validate Resident PIN
        pin = data.borrower["resident_address"]["pin"].value
        if pin:
            pin_digits = re.sub(r'\D', '', str(pin))
            if len(pin_digits) != 6:
                data.borrower["resident_address"]["pin"].is_valid = False
                data.borrower["resident_address"]["pin"].validation_note = "PIN code must be 6 digits"
            else:
                data.borrower["resident_address"]["pin"].value = pin_digits
