"""
Multi-Document Data Consolidation and Conflict Handling Module.
WBGB Housing Loan Processing System.

Key Responsibilities:
1. Multi-document aggregation into NormalizedLoanData structure.
2. Contextual Person Role Identification (Borrower, Co-Borrower, Guarantor 1, Guarantor 2, Property Owner, Other).
3. Value comparison across documents:
   - Identical/matching values: verified and consolidated with multiple source document references.
   - Differing values: flagged as conflicts with status NEEDS_MANUAL_REVIEW without silent overwriting.
   - Missing values: strictly preserved as None/null (zero hallucination).
4. Full Audit Trail generation for compliance and manual reviewer verification.
"""

import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from normalizer import NormalizedLoanData, NormalizedField, LABEL_ALIASES


# ==============================================================================
# ENUMS AND CONSTANTS
# ==============================================================================

class PersonRole:
    BORROWER = "borrower"
    CO_BORROWER = "co_borrower"
    GUARANTOR_1 = "guarantor_1"
    GUARANTOR_2 = "guarantor_2"
    PROPERTY_OWNER = "property_owner"
    OTHER = "other"


class ReviewStatus:
    VERIFIED = "VERIFIED_ACROSS_DOCUMENTS"  # Multiple sources agree on value
    SINGLE_SOURCE = "SINGLE_SOURCE"          # Single source, no conflict
    NEEDS_MANUAL_REVIEW = "NEEDS_MANUAL_REVIEW"  # Multiple sources disagree
    MISSING = "MISSING"                      # Field was not found in any uploaded doc


# ==============================================================================
# AUDIT TRAIL ENTRY
# ==============================================================================

class AuditEntry:
    """Represents a single field in the consolidation audit trail."""
    def __init__(
        self,
        field_path: str,
        label: str,
        entity_role: str,
        value: Optional[Any],
        review_status: str,
        has_conflict: bool = False,
        source_documents: Optional[List[str]] = None,
        conflicting_values: Optional[List[Dict[str, Any]]] = None,
        all_sources: Optional[List[Dict[str, Any]]] = None,
        notes: Optional[str] = None
    ):
        self.field_path = field_path
        self.label = label
        self.entity_role = entity_role
        self.value = value
        self.review_status = review_status
        self.has_conflict = has_conflict
        self.source_documents = source_documents or []
        self.conflicting_values = conflicting_values or []
        self.all_sources = all_sources or []
        self.notes = notes

    def to_dict(self) -> Dict[str, Any]:
        return {
            "field_path": self.field_path,
            "label": self.label,
            "entity_role": self.entity_role,
            "value": self.value,
            "review_status": self.review_status,
            "has_conflict": self.has_conflict,
            "source_documents": self.source_documents,
            "conflicting_values": self.conflicting_values,
            "all_sources": self.all_sources,
            "notes": self.notes
        }


# ==============================================================================
# VALUE NORMALIZER & COMPARATOR
# ==============================================================================

def normalize_compare_key(val: Any) -> str:
    """
    Normalizes a value for intelligent semantic comparison across OCR results:
    - Removes whitespace and punctuation formatting
    - Lowercases text
    - Normalizes dates (DD/MM/YYYY vs DD.MM.YYYY vs DD-MM-YYYY)
    - Normalizes currency/numbers (strips Rs, commas, trailing .00)
    """
    if val is None:
        return ""
    s = str(val).strip().lower()

    # Remove extra internal spaces
    s = re.sub(r'\s+', ' ', s)

    # Date normalization: DD.MM.YYYY, DD-MM-YYYY, DD/MM/YYYY -> DD/MM/YYYY
    date_m = re.match(r'^(\d{1,2})[\.\-\/](\d{1,2})[\.\-\/](\d{2,4})$', s)
    if date_m:
        d, m, y = date_m.group(1).zfill(2), date_m.group(2).zfill(2), date_m.group(3)
        return f"{d}/{m}/{y}"

    # Number / Currency normalization: 2500000 == 25,00,000 == Rs. 2500000
    if re.search(r'\d', s) and not re.search(r'[a-zA-Z]{3,}', s):
        clean_num = re.sub(r'[^\d.]', '', s)
        try:
            f = float(clean_num)
            return str(int(f)) if f.is_integer() else str(f)
        except ValueError:
            pass

    # Phone / Aadhaar normalization (digits only)
    if re.match(r'^[\d\s\-\+]{10,14}$', s):
        digits = re.sub(r'\D', '', s)
        if len(digits) > 10 and digits.startswith("91"):
            digits = digits[2:]
        return digits

    # General text: strip leading/trailing punctuation
    s = re.sub(r'^[,\.\-\:\s]+|[,\.\-\:\s]+$', '', s)
    return s


def values_match(val1: Any, val2: Any) -> bool:
    """Checks if two extracted values represent the same information."""
    k1 = normalize_compare_key(val1)
    k2 = normalize_compare_key(val2)
    if not k1 or not k2:
        return False
    return k1 == k2


# ==============================================================================
# CONTEXTUAL ROLE RESOLVER
# ==============================================================================

class ContextualRoleResolver:
    """
    Determines the legal/application role of each document and person:
    - Analyzes document type, filename, headings, explicit role labels, and relationships.
    - Anchors master identities (e.g. Rahul Sharma as Borrower, Amit Sharma as Guarantor).
    - Prevents guessing based only on isolated person names.
    """

    def __init__(self):
        self.borrower_names: Set[str] = set()
        self.guarantor_1_names: Set[str] = set()
        self.guarantor_2_names: Set[str] = set()
        self.property_owner_names: Set[str] = set()

    def discover_anchors_from_context(self, documents: List[Dict[str, Any]]):
        """
        Phase 1: Scans master documents (Loan applications, Sanction forms, Deeds)
        to discover explicitly declared person-to-role relationships.
        """
        for doc in documents:
            doc_name = doc.get("document_name", "").lower()
            doc_type = doc.get("document_type", "").lower()
            fields = doc.get("fields", {})
            ocr_text = doc.get("ocr_text", "").lower()

            # 1. Master Loan Application or KYC Form declarations
            if any(k in doc_name or k in doc_type for k in ["application", "sanction", "loan", "kyc"]):
                # Check for explicit borrower labels
                for label, val in fields.items():
                    l_lower = label.lower()
                    if val and isinstance(val, str) and len(val.strip()) > 2:
                        if any(b in l_lower for b in ["borrower", "applicant name", "name of applicant", "full name"]):
                            if not any(g in l_lower for g in ["guarantor", "co-"]):
                                self.borrower_names.add(normalize_compare_key(val))
                        elif "guarantor 1" in l_lower or "first guarantor" in l_lower:
                            self.guarantor_1_names.add(normalize_compare_key(val))
                        elif "guarantor 2" in l_lower or "second guarantor" in l_lower:
                            self.guarantor_2_names.add(normalize_compare_key(val))
                        elif "guarantor" in l_lower or "surety" in l_lower:
                            if not self.guarantor_1_names:
                                self.guarantor_1_names.add(normalize_compare_key(val))
                            else:
                                self.guarantor_2_names.add(normalize_compare_key(val))

            # 2. Guarantee Deed or Letter of Guarantee declarations
            if any(k in doc_name or k in doc_type for k in ["guarantee", "surety"]):
                for label, val in fields.items():
                    l_lower = label.lower()
                    if val and isinstance(val, str) and len(val.strip()) > 2:
                        if any(g in l_lower for g in ["guarantor name", "name of guarantor", "full name", "name"]):
                            if "2" in doc_name or "second" in doc_name:
                                self.guarantor_2_names.add(normalize_compare_key(val))
                            else:
                                self.guarantor_1_names.add(normalize_compare_key(val))

            # 3. Property Deed declarations (Sale Deed, Gift Deed, Partition)
            if any(k in doc_name or k in doc_type for k in ["deed", "sale deed", "property", "title"]):
                for label, val in fields.items():
                    l_lower = label.lower()
                    if val and isinstance(val, str) and len(val.strip()) > 2:
                        if any(p in l_lower for p in ["owner", "purchaser", "buyer", "vendee"]):
                            self.property_owner_names.add(normalize_compare_key(val))

    def resolve_document_role(self, doc: Dict[str, Any]) -> str:
        """
        Phase 2: Classifies a document's target entity role using:
        1. Contextual filename markers (e.g. 'guarantor1_pan.pdf', 'borrower_aadhaar.pdf')
        2. Document classification (e.g. 'Sale Deed' -> property, 'Salary Slip' -> borrower)
        3. Person name matching against established anchors.
        """
        doc_name = doc.get("document_name", "").lower()
        doc_type = doc.get("document_type", "").lower()
        fields = doc.get("fields", {})
        ocr_text = doc.get("ocr_text", "").lower()

        # Step A: Contextual Filename and Classification Clues
        if any(k in doc_name for k in ["guarantor_2", "guarantor 2", "guarantor2", "g2"]):
            return PersonRole.GUARANTOR_2
        if any(k in doc_name for k in ["guarantor_1", "guarantor 1", "guarantor1", "g1", "guarantor"]):
            return PersonRole.GUARANTOR_1
        if any(k in doc_name or k in doc_type for k in ["letter of guarantee", "guarantee deed", "surety"]):
            return PersonRole.GUARANTOR_2 if "2" in doc_name else PersonRole.GUARANTOR_1

        if any(k in doc_name or k in doc_type for k in ["deed", "sale deed", "khatian", "parcha", "search report", "valuation"]):
            return PersonRole.PROPERTY_OWNER

        if any(k in doc_name or k in doc_type for k in ["salary", "payslip", "pay slip", "form 16"]):
            return PersonRole.BORROWER

        if any(k in doc_name for k in ["co_borrower", "co-borrower", "co_applicant", "co-applicant"]):
            return PersonRole.CO_BORROWER

        if any(k in doc_name for k in ["borrower", "applicant"]):
            return PersonRole.BORROWER

        # Step B: Check person name extracted from identity documents (PAN, Aadhaar, Voter, DL)
        name_val = None
        for k, v in fields.items():
            if any(nk in k.lower() for nk in ["name", "full name", "holder's name", "customer name"]):
                name_val = v
                break

        if name_val:
            comp_key = normalize_compare_key(name_val)
            if comp_key in self.guarantor_2_names:
                return PersonRole.GUARANTOR_2
            if comp_key in self.guarantor_1_names:
                return PersonRole.GUARANTOR_1
            if comp_key in self.borrower_names:
                return PersonRole.BORROWER
            if comp_key in self.property_owner_names:
                return PersonRole.PROPERTY_OWNER

        # Step C: Default fallback for general identity/application docs is Borrower
        return PersonRole.BORROWER


# ==============================================================================
# CONSOLIDATION AND CONFLICT HANDLING ENGINE
# ==============================================================================

class MultiDocumentConsolidator:
    """
    Core engine for multi-document consolidation.
    Aggregates fields, performs semantic matching, detects conflicts,
    preserves full provenance, and generates an audit trail.
    """

    def __init__(self):
        # Inverted index for label aliases
        self.alias_index: Dict[str, str] = {}
        for canonical, aliases in LABEL_ALIASES.items():
            for alias in aliases:
                self.alias_index[alias.strip().lower()] = canonical

    def consolidate(
        self,
        extraction_payload: Union[Dict[str, Any], List[Dict[str, Any]]],
        explicit_role_overrides: Optional[Dict[str, str]] = None
    ) -> Tuple[NormalizedLoanData, List[AuditEntry], Dict[str, Any]]:
        """
        Main consolidation routine.

        Returns:
        1. normalized_loan_data: Populated NormalizedLoanData container
        2. audit_trail: List of AuditEntry objects detailing every field
        3. consolidation_summary: High-level metrics (total, verified, conflicts, missing)
        """
        role_overrides = explicit_role_overrides or {}

        # 1. Parse documents list
        documents = []
        engine_used = None
        if isinstance(extraction_payload, dict):
            documents = extraction_payload.get("documents", [])
            engine_used = extraction_payload.get("engine_used")
        elif isinstance(extraction_payload, list):
            documents = extraction_payload

        # 2. Discover context anchors (borrower vs guarantor names)
        resolver = ContextualRoleResolver()
        resolver.discover_anchors_from_context(documents)

        # 3. Create fresh normalized container
        normalized_data = NormalizedLoanData()
        normalized_data.other_info["engine_used"] = engine_used
        normalized_data.other_info["processed_count"] = len(documents)

        # Intermediate field collector:
        # { canonical_path: [ { value, raw_value, source_doc, confidence, evidence, entity_role } ] }
        field_observations: Dict[str, List[Dict[str, Any]]] = {}

        # 4. Ingest and route each document
        for doc in documents:
            doc_name = doc.get("document_name", "Document")
            doc_type = doc.get("document_type", "Document")
            fields = doc.get("fields", {})
            ocr_text = doc.get("ocr_text", "")
            confidence = doc.get("confidence", 0.90)

            normalized_data.other_info["source_documents"].append({
                "name": doc_name,
                "type": doc_type,
                "field_count": len(fields)
            })

            # Determine entity role for this document
            entity_role = role_overrides.get(doc_name) or resolver.resolve_document_role(doc)

            items = fields.items() if isinstance(fields, dict) else []
            for raw_label, raw_val in items:
                if raw_val is None or str(raw_val).strip() == "":
                    continue

                clean_label = str(raw_label).strip().lower()
                canonical_path = self.alias_index.get(clean_label)

                evidence_snippet = self._find_evidence_line(ocr_text, str(raw_val))

                if canonical_path:
                    final_path = self._route_path_by_role(canonical_path, entity_role)
                    if final_path not in field_observations:
                        field_observations[final_path] = []

                    field_observations[final_path].append({
                        "value": raw_val,
                        "raw_value": raw_val,
                        "source_document": doc_name,
                        "confidence": confidence,
                        "evidence": evidence_snippet,
                        "entity_role": entity_role
                    })
                else:
                    normalized_data.other_info["unmapped_fields"].append({
                        "label": raw_label,
                        "value": raw_val,
                        "source_document": doc_name,
                        "entity_role": entity_role,
                        "evidence": evidence_snippet
                    })

        # 5. Consolidate Observations & Analyze Conflicts
        audit_trail: List[AuditEntry] = []
        conflict_count = 0
        verified_count = 0
        single_source_count = 0

        for path, observations in field_observations.items():
            if not observations:
                continue

            # Compare all observed values
            unique_normalized_values: Dict[str, List[Dict[str, Any]]] = {}
            for obs in observations:
                norm_key = normalize_compare_key(obs["value"])
                if norm_key not in unique_normalized_values:
                    unique_normalized_values[norm_key] = []
                unique_normalized_values[norm_key].append(obs)

            has_conflict = len(unique_normalized_values) > 1
            all_source_docs = list(dict.fromkeys(obs["source_document"] for obs in observations))
            entity_role = observations[0].get("entity_role", "borrower")

            if has_conflict:
                conflict_count += 1
                review_status = ReviewStatus.NEEDS_MANUAL_REVIEW
                
                # Format conflict details
                conflicting_items = []
                for nkey, group in unique_normalized_values.items():
                    sample = group[0]
                    conflicting_items.append({
                        "value": sample["value"],
                        "sources": [g["source_document"] for g in group],
                        "evidence": [g["evidence"] for g in group if g["evidence"]]
                    })

                # Sort by confidence or completeness for recommendation
                best_obs = max(observations, key=lambda x: (x.get("confidence") or 0.0, len(str(x.get("value") or ""))))
                primary_val = best_obs["value"]
                notes = f"Conflict detected between {len(unique_normalized_values)} distinct values across {len(all_source_docs)} documents. Recommended: '{primary_val}'"

            elif len(observations) > 1:
                verified_count += 1
                review_status = ReviewStatus.VERIFIED
                conflicting_items = []
                primary_val = observations[0]["value"]
                notes = f"Confirmed consistent across {len(all_source_docs)} documents: {', '.join(all_source_docs)}"

            else:
                single_source_count += 1
                review_status = ReviewStatus.SINGLE_SOURCE
                conflicting_items = []
                primary_val = observations[0]["value"]
                notes = f"Extracted from single source: {all_source_docs[0]}"

            # Write consolidated value into normalized data container
            normalized_data.set_by_path(
                path,
                value=primary_val,
                source_doc=", ".join(all_source_docs),
                confidence=observations[0].get("confidence"),
                evidence=observations[0].get("evidence")
            )

            # Mark conflict flag on the specific field object
            field_obj = self._get_field_by_path(normalized_data, path)
            if field_obj:
                field_obj.has_conflict = has_conflict
                field_obj.history = observations

            # Add to audit trail
            label = path.split(".")[-1].replace("_", " ").title()
            audit_trail.append(AuditEntry(
                field_path=path,
                label=label,
                entity_role=entity_role,
                value=primary_val,
                review_status=review_status,
                has_conflict=has_conflict,
                source_documents=all_source_docs,
                conflicting_values=conflicting_items,
                all_sources=observations,
                notes=notes
            ))

        # 6. Audit unextracted / missing fields
        flat_all = normalized_data.to_flat_values()
        missing_count = 0
        for fpath, fval in flat_all.items():
            if fval is None and fpath not in field_observations:
                missing_count += 1
                label = fpath.split(".")[-1].replace("_", " ").title()
                role = "guarantor" if "guarantor" in fpath else ("property" if "property" in fpath else "borrower")
                audit_trail.append(AuditEntry(
                    field_path=fpath,
                    label=label,
                    entity_role=role,
                    value=None,
                    review_status=ReviewStatus.MISSING,
                    has_conflict=False,
                    source_documents=[],
                    conflicting_values=[],
                    notes="Not present in any uploaded document."
                ))

        # 7. Post-process derivations (Names decomposition, Dates decomposition)
        self._post_process_derivatives(normalized_data)

        # 8. Summary statistics
        consolidation_summary = {
            "total_documents_processed": len(documents),
            "total_fields_extracted": len(field_observations),
            "verified_multi_source_fields": verified_count,
            "single_source_fields": single_source_count,
            "conflicting_fields": conflict_count,
            "missing_fields": missing_count,
            "unmapped_fields": len(normalized_data.other_info["unmapped_fields"]),
            "timestamp": datetime.now().isoformat()
        }

        return normalized_data, audit_trail, consolidation_summary

    def _route_path_by_role(self, canonical_path: str, entity_role: str) -> str:
        """Reroutes a canonical path to the correct person section based on entity role."""
        if entity_role == PersonRole.GUARANTOR_1:
            if canonical_path.startswith("borrower."):
                return f"guarantors.0.{canonical_path[len('borrower.'):]}"
            if canonical_path.startswith("guarantor."):
                return f"guarantors.0.{canonical_path[len('guarantor.'):]}"
        elif entity_role == PersonRole.GUARANTOR_2:
            if canonical_path.startswith("borrower."):
                return f"guarantors.1.{canonical_path[len('borrower.'):]}"
            if canonical_path.startswith("guarantor."):
                return f"guarantors.1.{canonical_path[len('guarantor.'):]}"
        elif entity_role == PersonRole.PROPERTY_OWNER:
            if canonical_path.startswith("borrower.resident_address"):
                return canonical_path.replace("borrower.resident_address", "property")
        return canonical_path

    def _find_evidence_line(self, ocr_text: str, val_str: str) -> Optional[str]:
        """Finds matching context line in OCR text."""
        if not ocr_text or not val_str or len(val_str) < 3:
            return None
        c_val = val_str.strip().lower()
        for line in ocr_text.splitlines():
            if c_val in line.lower():
                return line.strip()[:140]
        return None

    def _get_field_by_path(self, data: NormalizedLoanData, path: str) -> Optional[NormalizedField]:
        """Traverses NormalizedLoanData to retrieve the NormalizedField instance."""
        parts = path.split(".")
        curr: Any = data
        for p in parts:
            if hasattr(curr, p):
                curr = getattr(curr, p)
            elif isinstance(curr, dict) and p in curr:
                curr = curr[p]
            elif isinstance(curr, list) and p.isdigit() and int(p) < len(curr):
                curr = curr[int(p)]
            else:
                return None
        return curr if isinstance(curr, NormalizedField) else None

    def _post_process_derivatives(self, data: NormalizedLoanData):
        """Derives split names and date components."""
        # Borrower Name decomposition
        b_name = data.borrower["full_name"].value
        if b_name and not data.borrower["first_name"].value:
            parts = str(b_name).strip().split()
            title = ""
            if parts and parts[0].upper() in ["MR", "MR.", "MRS", "MRS.", "MS", "MS.", "SHRI", "SMT", "MD", "DR"]:
                title = parts[0]
                parts = parts[1:]
            if len(parts) == 1:
                first = parts[0]
                sur = ""
            elif len(parts) >= 2:
                first = " ".join(parts[:-1])
                sur = parts[-1]
            else:
                first, sur = "", ""
            if title: data.borrower["title"].update(title, source_document="Derived from Full Name", confidence=1.0)
            if first: data.borrower["first_name"].update(first, source_document="Derived from Full Name", confidence=1.0)
            if sur: data.borrower["surname"].update(sur, source_document="Derived from Full Name", confidence=1.0)

        # Date decomposition
        app_date = data.loan["application_date"].value
        if app_date and not data.loan["application_day"].value:
            m = re.search(r'(\d{1,2})[\/\.\-](\d{1,2})[\/\.\-](\d{4})', str(app_date))
            if m:
                d_int = int(m.group(1))
                y_int = int(m.group(3))
                suffix = 'th' if 11 <= (d_int % 100) <= 13 else {1: 'st', 2: 'nd', 3: 'rd'}.get(d_int % 10, 'th')
                day_str = f"{d_int}{suffix}"
                month_names = {
                    1: "january", 2: "february", 3: "march", 4: "april", 5: "may", 6: "june",
                    7: "july", 8: "august", 9: "september", 10: "october", 11: "november", 12: "december"
                }
                m_int = int(m.group(2))
                month_str = month_names.get(m_int, "")
                data.loan["application_day"].update(day_str, source_document="Derived from Date", confidence=1.0)
                data.loan["application_month"].update(month_str, source_document="Derived from Date", confidence=1.0)
                data.loan["application_year"].update(y_int, source_document="Derived from Date", confidence=1.0)
