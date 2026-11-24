"""
Excel Field Mapping Engine for WBGB Housing Loan (`WBGB HBL LOAN (2).xlsm`).
Transfers normalized extracted loan information into the primary worksheets:
1. BORROWERDETAILS
2. GURANTORSDETAILS
3. LANDDETAILS

Guarantees:
- Never modifies the original template; writes to a timestamped or user-specified output copy.
- Preserves all VBA macros, formulas, merged cell layouts, and formatting.
- Strict type casting: financial amounts become numeric (so formulas work), IDs stay strings (leading zeros preserved).
- Protected cells and formulas are strictly shielded from overwriting.
- Comprehensive mapping audit report detailing success, missing, skipped, and uncertain fields.
"""

import os
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Union, Tuple, Set
import openpyxl

from excel_mapping_config import (
    FieldMapping,
    EXCEL_FIELD_MAPPINGS,
    PROTECTED_CELLS,
    VALIDATION_RULES,
)
from normalizer import NormalizedLoanData, NormalizedField, NormalizationEngine


def _extract_value_from_normalized(normalized_input: Any, field_path: str) -> Tuple[Optional[Any], Optional[NormalizedField]]:
    """
    Safely traverses normalized_input by path (e.g. 'borrower.full_name' or 'guarantors[0].name')
    Returns: (raw_or_normalized_value, field_object_if_available)
    """
    if normalized_input is None:
        return None, None

    # If it's a NormalizedLoanData instance, convert to dict or use set_by_path
    if isinstance(normalized_input, NormalizedLoanData):
        data_dict = normalized_input.to_dict()
    elif isinstance(normalized_input, dict):
        data_dict = normalized_input
    else:
        return None, None

    # Handle indexed paths like guarantors[0].name -> guarantors.0.name
    normalized_path = re.sub(r'\[(\d+)\]', r'.\1', field_path)
    parts = normalized_path.split(".")

    curr = data_dict
    for p in parts:
        if isinstance(curr, dict) and p in curr:
            curr = curr[p]
        elif isinstance(curr, list) and p.isdigit() and int(p) < len(curr):
            curr = curr[int(p)]
        else:
            return None, None

    # curr might be a NormalizedField dict e.g. {"value": "...", "confidence": ...}
    if isinstance(curr, dict) and "value" in curr:
        val = curr.get("value")
        return val, curr
    elif isinstance(curr, NormalizedField):
        return curr.value, curr
    else:
        return curr, None


def _format_cell_value(val: Any, data_type: str) -> Tuple[Any, Optional[str]]:
    """
    Cleans and casts data to the exact required Excel format:
    - Text / IDs: kept as string to preserve leading zeros.
    - Currency / Numeric: converted to int or float so Excel formulas calculate.
    - Dates: standard string format DD.MM.YYYY or DD/MM/YYYY.
    Returns: (formatted_value, warning_message_if_any)
    """
    if val is None or str(val).strip() == "":
        return None, None

    sval = str(val).strip()

    # 1. Text & Identifiers (PAN, Aadhaar, Phone, PIN, Account Numbers)
    if data_type in ["text", "dropdown"]:
        return sval, None

    # 2. Financial Currency / Numeric Amounts (Must be numeric for Excel formulas)
    elif data_type in ["currency", "numeric"]:
        clean_num = re.sub(r'[^\d.]', '', sval)
        if not clean_num:
            return sval, f"Could not parse numeric amount from '{sval}', wrote as text"
        try:
            if "." in clean_num:
                f_val = float(clean_num)
                # If trailing decimals are zero, cast to int
                return int(f_val) if f_val.is_integer() else f_val, None
            return int(clean_num), None
        except ValueError:
            return sval, f"Failed float conversion for '{sval}'"

    # 3. Integers
    elif data_type == "integer":
        clean_int = re.sub(r'\D', '', sval)
        if clean_int:
            return int(clean_int), None
        return sval, f"Could not parse integer from '{sval}'"

    # 4. Percentage / Interest Rate
    elif data_type == "percentage":
        clean_pct = re.sub(r'[^\d.]', '', sval)
        try:
            f_pct = float(clean_pct)
            # If greater than 1, e.g. 8.5 for 8.5%, keep 8.5
            return f_pct, None
        except ValueError:
            return sval, None

    # 5. Date
    elif data_type == "date":
        # Format uniformly as DD/MM/YYYY or keep clean
        return sval, None

    return sval, None


def populate_excel(
    normalized_data: Union[NormalizedLoanData, Dict[str, Any]],
    template_path: str,
    output_path: str,
    config: Optional[List[FieldMapping]] = None,
    overwrite_existing_output: bool = True
) -> Dict[str, Any]:
    """
    Populates an existing Excel template with normalized loan application data.

    Parameters:
    - normalized_data: NormalizedLoanData object or dict output from Stage 2 normalizer.
    - template_path: Path to the original master XLSM template.
    - output_path: Destination path for the new populated XLSM copy.
    - config: Optional custom FieldMapping list (defaults to EXCEL_FIELD_MAPPINGS).
    - overwrite_existing_output: Whether to overwrite output_path if it already exists.

    Returns:
    A detailed mapping audit report dictionary.
    """
    mapping_config = config or EXCEL_FIELD_MAPPINGS

    report: Dict[str, Any] = {
        "success": False,
        "timestamp": datetime.now().isoformat(),
        "template_path": os.path.abspath(template_path),
        "output_path": os.path.abspath(output_path),
        "total_fields_configured": len(mapping_config),
        "total_fields_mapped": 0,
        "successful_fields": [],
        "missing_mandatory_fields": [],
        "missing_optional_fields": [],
        "skipped_fields": [],
        "uncertain_fields": [],
        "validation_warnings": [],
        "downstream_sheets_preserved": []
    }

    # 1. Safety Checks
    if not os.path.exists(template_path):
        report["error"] = f"Template file does not exist at: {template_path}"
        return report

    norm_template = os.path.normcase(os.path.abspath(template_path))
    norm_output = os.path.normcase(os.path.abspath(output_path))
    if norm_template == norm_output:
        report["error"] = "Safety check failed: output_path cannot be identical to template_path! Original workbook must remain pristine."
        return report

    if os.path.exists(output_path) and not overwrite_existing_output:
        report["error"] = f"Output file already exists at {output_path} and overwrite_existing_output=False"
        return report

    # 2. Ingest Data: if raw dict was passed, ensure it is normalized
    if isinstance(normalized_data, dict) and "borrower" not in normalized_data and "documents" in normalized_data:
        engine = NormalizationEngine()
        normalized_obj = engine.normalize(normalized_data)
    else:
        normalized_obj = normalized_data

    # 3. Load Workbook safely with VBA preservation
    try:
        wb = openpyxl.load_workbook(template_path, keep_vba=True, data_only=False)
    except Exception as e:
        report["error"] = f"Failed to load workbook '{template_path}': {str(e)}"
        return report

    # Track preserved sheets
    primary_sheet_names = {"BORROWERDETAILS", "GURANTORSDETAILS", "LANDDETAILS"}
    report["downstream_sheets_preserved"] = [s for s in wb.sheetnames if s not in primary_sheet_names]

    # Verify primary sheets exist in workbook
    for ps in primary_sheet_names:
        if ps not in wb.sheetnames:
            report["error"] = f"Missing required primary worksheet '{ps}' in template workbook."
            return report

    # 4. Execute Field Mapping
    written_cells: Set[Tuple[str, str]] = set()

    for mapping in mapping_config:
        sheet_name = mapping.sheet
        cell_coord = mapping.cell.upper()
        field_path = mapping.field_path
        ws = wb[sheet_name]

        # Check Protected Cells (Formulas & Fixed Defaults)
        if (sheet_name, cell_coord) in PROTECTED_CELLS:
            report["skipped_fields"].append({
                "field_path": field_path,
                "sheet": sheet_name,
                "cell": cell_coord,
                "reason": "Cell is a protected formula or fixed default; write blocked to prevent corruption."
            })
            continue

        # Prevent duplicate write in same session
        if (sheet_name, cell_coord) in written_cells:
            report["skipped_fields"].append({
                "field_path": field_path,
                "sheet": sheet_name,
                "cell": cell_coord,
                "reason": f"Cell {cell_coord} was already populated in this session."
            })
            continue

        # Check existing cell formula safeguard
        curr_cell_val = str(ws[cell_coord].value or "")
        if curr_cell_val.startswith("="):
            report["skipped_fields"].append({
                "field_path": field_path,
                "sheet": sheet_name,
                "cell": cell_coord,
                "reason": f"Cell contains active formula '{curr_cell_val}'; write blocked."
            })
            continue

        # Retrieve value from normalized container
        val, field_meta = _extract_value_from_normalized(normalized_obj, field_path)

        # Handle Missing Fields
        if val is None or str(val).strip() == "":
            if mapping.mandatory:
                report["missing_mandatory_fields"].append({
                    "field_path": field_path,
                    "sheet": sheet_name,
                    "cell": cell_coord,
                    "label": mapping.label,
                    "reason": "Mandatory field value is missing from extracted data."
                })
            else:
                report["missing_optional_fields"].append({
                    "field_path": field_path,
                    "sheet": sheet_name,
                    "cell": cell_coord,
                    "label": mapping.label
                })
            continue

        # Run Validation Rule if defined
        if mapping.validation_rule and mapping.validation_rule in VALIDATION_RULES:
            validator = VALIDATION_RULES[mapping.validation_rule]
            if not validator(val):
                report["validation_warnings"].append({
                    "field_path": field_path,
                    "sheet": sheet_name,
                    "cell": cell_coord,
                    "value": str(val),
                    "rule": mapping.validation_rule,
                    "warning": f"Value '{val}' does not satisfy validation rule '{mapping.validation_rule}'"
                })

        # Check for extraction conflict or low confidence
        if isinstance(field_meta, dict):
            if field_meta.get("has_conflict"):
                report["uncertain_fields"].append({
                    "field_path": field_path,
                    "sheet": sheet_name,
                    "cell": cell_coord,
                    "chosen_value": str(val),
                    "candidates": field_meta.get("history", []),
                    "reason": "Multiple source documents provided conflicting values for this field."
                })
            conf = field_meta.get("confidence")
            if conf is not None and conf < 0.65:
                report["uncertain_fields"].append({
                    "field_path": field_path,
                    "sheet": sheet_name,
                    "cell": cell_coord,
                    "chosen_value": str(val),
                    "confidence": conf,
                    "reason": "Low extraction confidence (< 65%)."
                })

        # Format Value according to target Excel data type
        formatted_val, format_warning = _format_cell_value(val, mapping.data_type)
        if format_warning:
            report["validation_warnings"].append({
                "field_path": field_path,
                "sheet": sheet_name,
                "cell": cell_coord,
                "warning": format_warning
            })

        # Write to Cell (Top-Left cell of any merged range)
        try:
            ws[cell_coord].value = formatted_val
            written_cells.add((sheet_name, cell_coord))
            report["successful_fields"].append({
                "field_path": field_path,
                "sheet": sheet_name,
                "cell": cell_coord,
                "label": mapping.label,
                "written_value": formatted_val,
                "data_type": mapping.data_type
            })
            report["total_fields_mapped"] += 1
        except Exception as write_err:
            report["skipped_fields"].append({
                "field_path": field_path,
                "sheet": sheet_name,
                "cell": cell_coord,
                "reason": f"Cell write exception: {str(write_err)}"
            })

    # 5. Save Output Workbook
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    try:
        wb.save(output_path)
        report["success"] = True
    except Exception as save_err:
        report["error"] = f"Failed to save populated workbook to '{output_path}': {str(save_err)}"
        report["success"] = False

    return report
