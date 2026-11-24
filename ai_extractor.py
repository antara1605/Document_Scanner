import os
import json
import re
from dotenv import load_dotenv
from groq import Groq

load_dotenv(os.path.join(os.path.abspath(os.path.dirname(__file__)), ".env"))

SYSTEM_PROMPT = """You are an expert document information extraction AI.
Your job is to analyze OCR text extracted from uploaded documents and extract ALL meaningful structured information deterministically and consistently.

CRITICAL INSTRUCTIONS:
1. ONLY extract information that is explicitly present in the document text.
2. NEVER invent, guess, or hallucinate missing information.
3. DO NOT extract empty, null, 'None', 'N/A', 'NIL', or placeholder dashes. ONLY include fields that contain actual document information.
4. Extract distinct individual data points consistently:
   - Personal Details: Full Name, Father/Spouse Name, Date of Birth, Gender, Age, Marital Status
   - Driving License: License Number, Current Status, Holder's Name, DOB, Issue Date, Validity, Issuing Office, Vehicle Classes (COV: LMV, MCWG)
   - Banking / Cheque: Bank Name, Branch Name, Account Number, IFSC Code, Cheque Number, Account Type
   - Identity Numbers: PAN Number, Aadhaar Number, Voter ID, Passport Number, License Number
   - Contact / Address: Mobile Number, Email, Address, Village/Town, District, State, PIN Code
   - Property / Legal: Deed Number, Plot/Dag No, Khatian No, Mouza, Area, Sub-Registry Office
   - Financial / Income: Salary, Amounts, Rates, Deductions, Balances, Prices
   - Survey / Insurance / Vehicle: Policy Number, Claim Number, Vehicle Registration No, Chassis No, Engine No, Make/Model, Date of Loss, Surveyor Name, License No
   - Dates: Dates of issue, validity, registration, agreement, inspection
5. Return clean, human-readable field labels (e.g. "Full Name", "Date of Birth", "Bank Name", "Account Number", "IFSC Code", "PAN Number", etc.)
6. Format dates as DD/MM/YYYY or as they clearly appear.
7. For each document, identify its type/category (e.g. "Driving License", "Aadhaar Card", "PAN Card", "Bank Cheque", "Salary Slip", "Motor Survey Report", "Discharge Voucher", "Sale Deed", "Payment Receipt", "Tax Invoice", "Certificate", etc.)

You MUST return a JSON object with this exact schema:
{
  "documents": [
    {
      "document_name": "filename",
      "document_type": "detected type of document",
      "fields": {
        "Field Label": "extracted value",
        "Another Field": "another value"
      }
    }
  ]
}
"""

class AIExtractor:
    def __init__(self, api_key=None, model=None):
        self.api_key = api_key or os.getenv("GROQ_API_KEY", "")
        self.model = model or os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
        self.client = None
        if self.api_key and self.api_key.strip():
            try:
                self.client = Groq(api_key=self.api_key.strip())
            except Exception as e:
                print(f"Warning: Could not initialize Groq client: {e}")

    def extract_from_documents(self, documents):
        """
        documents: list of dicts:
        [
            {
                "id": int,
                "name": str,
                "category": str,
                "ocr_text": str,
                "pages": [{"page_number": int, "text": str}]
            },
            ...
        ]
        """
        has_text = any(bool(d.get("ocr_text", "").strip()) for d in documents)
        if not has_text:
            return {
                "engine_used": "None (No Text Detected)",
                "documents": [
                    {
                        "document_name": d.get("name", "Document"),
                        "document_type": "Unreadable / Empty Document",
                        "fields": {},
                        "summary": "No readable text detected in this document."
                    }
                    for d in documents
                ]
            }

        # Multi-page packet decomposition:
        # Detect multi-document packets in multi-page uploads
        target_docs = []
        for doc in documents:
            pages = doc.get("pages", [])
            doc_name = doc.get("name", "Document")
            if len(pages) > 1:
                for p in pages:
                    p_num = p.get("page_number", 1)
                    p_text = p.get("text", "").strip()
                    if not p_text:
                        continue
                    t_up = p_text.upper()
                    # Skip non-data index / guide pages
                    if "EXPECTED CROSS-DOCUMENT LINKS" in t_up or "TEST GUIDE" in t_up:
                        continue
                    if "WBGB HOUSING LOAN — FICTIONAL TEST DOCUMENT PACK" in t_up and "DOCUMENTS INCLUDED" in t_up:
                        continue

                    p_type = self.classify_document(p_text, f"{doc_name} Page {p_num}")
                    target_docs.append({
                        "id": f"{doc.get('id', 1)}_{p_num}",
                        "name": f"{doc_name} (Page {p_num})",
                        "parent_file": doc_name,
                        "page_number": p_num,
                        "category": p_type,
                        "ocr_text": p_text,
                        "pages": [p],
                        "confidence": p.get("confidence", 0.95)
                    })
            else:
                p_num = doc.get("pages", [{}])[0].get("page_number", 1) if doc.get("pages") else 1
                doc["page_number"] = p_num
                doc["parent_file"] = doc.get("name", "Document")
                target_docs.append(doc)

        if not target_docs:
            target_docs = documents

        if self.client:
            models_to_try = [
                self.model,
                "openai/gpt-oss-120b",
                "llama-3.3-70b-versatile",
                "llama-3.1-8b-instant"
            ]
            seen = set()
            for m in models_to_try:
                if not m or m in seen:
                    continue
                seen.add(m)
                try:
                    res = self._extract_with_groq(target_docs, model=m)
                    res["engine_used"] = f"Groq AI ({m})"
                    # Ensure each document has a summary string
                    for doc in res.get("documents", []):
                        d_type = doc.get("document_type", "Document")
                        d_fields = doc.get("fields", {})
                        if "summary" not in doc or not doc["summary"]:
                            doc["summary"] = self.generate_summary(d_type, d_fields)
                    return res
                except Exception as e:
                    print(f"Groq API call error with model {m}: {e}. Trying fallback...")

            print("All Groq models failed. Falling back to high-accuracy offline rule & heuristic engine.")
            res = self._extract_with_rules(target_docs)
            res["engine_used"] = "Local Rule & Heuristic Engine (Offline Fallback)"
            return res
        else:
            print("No Groq API client configured. Using high-accuracy Local Rule & Heuristic Engine (Offline).")
            res = self._extract_with_rules(target_docs)
            res["engine_used"] = "Local Rule & Heuristic Engine (Offline)"
            return res

    def _extract_with_groq(self, documents, model=None):
        target_model = model or self.model
        all_extracted_docs = []

        batch_size = 2
        for b_idx in range(0, len(documents), batch_size):
            chunk = documents[b_idx:b_idx + batch_size]
            doc_context_blocks = []
            for doc in chunk:
                doc_id = doc.get("id", "")
                doc_name = doc.get("name", "Document")
                p_num = doc.get("page_number", 1)
                ocr_text = doc.get("ocr_text", "")

                doc_block = [f"=== DOCUMENT: {doc_name} (Page {p_num}, ID: {doc_id}) ==="]
                doc_block.append(ocr_text.strip() if ocr_text.strip() else "[No text detected on this document]")
                doc_context_blocks.append("\n".join(doc_block))

            full_prompt = "\n\n".join(doc_context_blocks)

            response = self.client.chat.completions.create(
                model=target_model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"Extract all information from these documents:\n\n{full_prompt}"}
                ],
                response_format={"type": "json_object"},
                temperature=0.0,
                seed=42
            )

            content = response.choices[0].message.content or "{}"
            content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()
            if content.startswith('```'):
                content = re.sub(r'^```(?:json)?\s*', '', content)
                content = re.sub(r'\s*```$', '', content)
            content = content.strip()

            try:
                extracted_data = json.loads(content)
            except Exception:
                m = re.search(r'\{.*\}', content, re.DOTALL)
                extracted_data = json.loads(m.group(0)) if m else {}

            chunk_docs = []
            if isinstance(extracted_data, list):
                chunk_docs = extracted_data
            elif isinstance(extracted_data, dict):
                chunk_docs = extracted_data.get("documents", [])
                if not chunk_docs and "fields" in extracted_data:
                    chunk_docs = [extracted_data]

            for idx_c, cd in enumerate(chunk_docs):
                matching_in = chunk[idx_c] if idx_c < len(chunk) else chunk[-1]
                cd["document_name"] = matching_in.get("name", cd.get("document_name", "Document"))
                cd["parent_file"] = matching_in.get("parent_file", matching_in.get("name"))
                cd["page_number"] = matching_in.get("page_number", 1)
                cd["ocr_text"] = matching_in.get("ocr_text", "")
                all_extracted_docs.append(cd)

        return {"documents": all_extracted_docs}

    def classify_document(self, text, doc_name=""):
        """Classifies document by examining text content and file name."""
        t_up = (text + " " + doc_name).upper()

        if any(k in t_up for k in ["SANCTION ADVICE", "LOAN SANCTION", "SANCTION OF HOUSING LOAN", "SANCTION LETTER", "TERMS OF SANCTION"]):
            return "Loan Sanction Advice"
        elif any(k in t_up for k in ["GUARANTOR INCOME", "NET-WORTH DECLARATION", "NET WORTH DECLARATION", "GUARANTOR NET-WORTH", "STATEMENT OF ASSETS AND LIABILITIES OF GUARANTOR"]):
            return "Guarantor Income and Net-Worth"
        elif any(k in t_up for k in ["GUARANTOR IDENTITY", "GUARANTOR PROFILE", "GUARANTOR KYC", "GUARANTOR AND ADDRESS PROFILE", "GUARANTOR DETAILS"]):
            return "Guarantor Profile"
        elif any(k in t_up for k in ["HOUSING LOAN REQUEST", "REQUEST FOR HOUSING LOAN", "LOAN REQUEST LETTER", "LOAN APPLICATION", "APPLICATION FOR HOUSING LOAN", "APPLICATION FOR HOME LOAN"]):
            return "Loan Application"
        elif any(k in t_up for k in ["PROPERTY VALUATION", "VALUATION AND SITE", "VALUATION REPORT", "SITE PARTICULARS", "ESTIMATED DISTRESS VALUE"]):
            return "Property Valuation Report"
        elif any(k in t_up for k in ["LEGAL SEARCH", "NON-ENCUMBRANCE", "SEARCH REPORT", "ENCUMBRANCE RESULT", "NON ENCUMBRANCE"]):
            return "Legal Search Report"
        elif any(k in t_up for k in ["PROPERTY TAX", "MUNICIPAL DUES", "TAX / MUNICIPAL DUES", "HOLDING TAX", "MUNICIPAL TAX RECEIPT"]):
            return "Property Tax Receipt"
        elif any(k in t_up for k in ["EMPLOYMENT AND SALARY CERTIFICATE", "EMPLOYMENT CERTIFICATE", "SERVICE CERTIFICATE", "SALARY CERTIFICATE"]):
            return "Employment and Salary Certificate"
        elif any(k in t_up for k in ["BORROWER IDENTITY PROFILE", "BORROWER KYC PROFILE", "BORROWER PROFILE"]):
            return "Borrower Identity Profile"
        elif any(k in t_up for k in ["MOTOR SURVEY REPORT", "SPOT SURVEY", "SURVEYOR & LOSS ASSESSOR", "FINAL SURVEY REPORT", "SLA-", "IIISLA"]):
            return "Motor Survey Report"
        elif any(k in t_up for k in ["DISCHARGE VOUCHER", "SATISFACTION VOUCHER", "SETTLEMENT VOUCHER"]):
            return "Discharge Voucher"
        elif any(k in t_up for k in ["PAYMENT RECEIPT", "RECEIPT NO", "RECEIVED AMOUNT", "PAYMENT_RECEIPT"]):
            return "Payment Receipt"
        elif any(k in t_up for k in ["TAX INVOICE", "COMMERCIAL INVOICE", "INVOICE NO", "BILL OF SUPPLY"]):
            return "Tax Invoice"
        elif any(k in t_up for k in ["TOWING BILL", "TOWING RECEIPT"]):
            return "Towing Bill"
        elif any(k in t_up for k in ["DRIVING LICENCE", "DRIVING LICENSE", "DETAILS OF DRIVING LICENSE", "SARATHI", "RCDLSTATUS", "PARIVAHAN", "DL_STATUS", "DL STATUS", "DLNUMBER", "DL NUMBER", "AUTHORISATION TO DRIVE"]):
            return "Driving License"
        elif any(k in t_up for k in ["CANCEL_CHECK", "CANCELLED CHEQUE"]) or (("CHEQUE" in t_up or "PAYABLE AT PAR" in t_up or "IFS CODE" in t_up) and "RECEIPT" not in t_up and "VOUCHER" not in t_up):
            return "Bank Cheque"
        elif any(k in t_up for k in ["AADHAAR", "UNIQUE IDENTIFICATION", "MERA AADHAAR", "UIDAI"]):
            return "Aadhaar Card"
        elif any(k in t_up for k in ["INCOME TAX DEPARTMENT", "PERMANENT ACCOUNT NUMBER"]) or (re.search(r'\b[A-Z]{5}[0-9]{4}[A-Z]{1}\b', text) and "PAN" in t_up):
            return "PAN Card"
        elif any(k in t_up for k in ["DEED OF SALE", "SALE DEED", "SUB-REGISTRY OFFICE", "SUB-REGISTRAR OFFICE", "REGISTRATION DIRECTORATE", "CHAIN DEED", "PROPERTY TITLE DEED", "TITLE DEED", "DEED ABSTRACT"]):
            return "Sale Deed"
        elif any(k in t_up for k in ["PARTNERSHIP DEED", "DEED OF PARTNERSHIP"]):
            return "Partnership Deed"
        elif any(k in t_up for k in ["SALARY SLIP", "PAYSLIP", "BASIC PAY", "GROSS SALARY", "PAY SLIP"]):
            return "Salary Slip"
        elif any(k in t_up for k in ["AFFIDAVIT", "SWORN STATEMENT", "AFFIDAVIT_NOS"]):
            return "Affidavit"
        elif any(k in t_up for k in ["GST REGISTRATION", "GOODS AND SERVICES TAX", "GST_REGISTRATION"]):
            return "GST Registration Certificate"
        elif any(k in t_up for k in ["KNOW YOUR CUSTOMER", "KYC FORM", "CUSTOMER INFORMATION FORM"]):
            return "KYC Form"
        elif any(k in t_up for k in ["STATEMENT OF ACCOUNT", "ACCOUNT STATEMENT", "PASSBOOK"]):
            return "Bank Statement"
        return "Official Document"

    def generate_summary(self, doc_type, fields):
        """Generates a concise key summary line for master summary tables."""
        if doc_type == "Driving License":
            parts = []
            if "Driving License Number" in fields and fields["Driving License Number"]:
                parts.append(f"License No: {fields['Driving License Number']}")
            if "Holder's Name" in fields and fields["Holder's Name"]:
                parts.append(f"Holder: {fields['Holder\'s Name']}")
            if "Current Status" in fields and fields["Current Status"]:
                parts.append(f"Status: {fields['Current Status']}")
            if "Date of Birth" in fields and fields["Date of Birth"]:
                parts.append(f"DOB: {fields['Date of Birth']}")
            if "Class of Vehicle (COV)" in fields and fields["Class of Vehicle (COV)"]:
                parts.append(f"Class: {fields['Class of Vehicle (COV)']}")
            return " | ".join(parts) if parts else "Driving License"
        elif doc_type == "Motor Survey Report":
            surveyor = fields.get("Surveyor Name", "")
            veh = fields.get("Vehicle Registration No", "")
            pol = fields.get("Policy Number", "")
            res = "Survey Report"
            if veh: res += f" for Vehicle: {veh}"
            if pol: res += f" | Policy: {pol}"
            if surveyor: res += f" | Surveyor: {surveyor}"
            return res
        elif doc_type == "Bank Cheque":
            bank = fields.get("Bank Name", "Bank")
            acc = fields.get("Account Number", "")
            ifsc = fields.get("IFSC Code", "")
            chq = fields.get("Cheque Number", "")
            return f"{bank} Cheque | A/c: {acc} | IFSC: {ifsc} | Cheque No: {chq}"
        elif doc_type == "PAN Card":
            pan = fields.get("PAN Number", "")
            name = fields.get("Full Name", "")
            dob = fields.get("Date of Birth", "")
            return f"PAN: {pan} | Name: {name} | DOB: {dob}"
        elif doc_type == "Aadhaar Card":
            adh = fields.get("Aadhaar Number", "")
            name = fields.get("Full Name", "")
            return f"Aadhaar: {adh} | Name: {name}"
        elif doc_type in ["Payment Receipt", "Tax Invoice", "Towing Bill"]:
            no = fields.get("Receipt / Invoice No", "")
            cust = fields.get("Customer Name", "")
            amt = fields.get("Amount", "")
            return f"{doc_type} #{no} | Customer: {cust} | Amount: {amt}"
        elif doc_type == "Discharge Voucher":
            pol = fields.get("Policy Number", "")
            claim = fields.get("Claim Number", "")
            amt = fields.get("Settlement Amount", "")
            return f"Discharge Voucher | Claim: {claim} | Policy: {pol} | Settled: {amt}"
        elif doc_type == "Sale Deed":
            deed = fields.get("Deed Number", "")
            purchaser = fields.get("Purchaser / Buyer", "")
            price = fields.get("Consideration Price", "")
            return f"Deed No: {deed} | Buyer: {purchaser} | Price: {price}"
        else:
            items = [f"{k}: {v}" for k, v in list(fields.items())[:3]]
            return " | ".join(items) if items else f"Processed {doc_type}"

    def _extract_with_rules(self, documents):
        """
        High-accuracy deterministic and heuristic offline extractor for Indian documents,
        driving licenses, motor survey reports, vouchers, receipts, identity proofs, deeds,
        and arbitrary structured forms.
        """
        result_docs = []

        for doc in documents:
            doc_name = doc.get("name", "Document")
            text = doc.get("ocr_text", "")
            fields = {}
            t_up = text.upper()
            doc_type = self.classify_document(text, doc_name)

            # 1. MOTOR SURVEY REPORT
            if doc_type == "Motor Survey Report":
                lines = [l.strip() for l in text.split('\n') if l.strip()]
                for l in lines[:5]:
                    if any(w in l.upper() for w in ["SINGH", "KUMAR", "SHARMA", "GUPTA", "ROY", "DAS", "VERMA", "PATEL", "JAIN", "DARSHAN"]) and "REPORT" not in l.upper():
                        fields['Surveyor Name'] = l
                        break

                m = re.search(r'License\s*No\.?\s*([A-Za-z0-9\/\-]+)', text, re.I)
                if m: fields['Surveyor License No'] = m.group(1).strip()

                m = re.search(r'Valid\s*Upto\s*[\–\-:]*\s*([0-3]?\d[\/-][0-1]?\d[\/-]\d{4})', text, re.I)
                if m: fields['License Valid Upto'] = m.group(1).strip()

                m = re.search(r'Ref\.?\s*No\.?\s*([A-Za-z0-9\/\-]+)', text, re.I)
                if m: fields['Reference Number'] = m.group(1).strip()

                m = re.search(r'POLICY\s*NO\.?\s*[:\s\n\r]*([A-Za-z0-9\/\-]+)', text, re.I)
                if m: fields['Policy Number'] = m.group(1).strip()

                m = re.search(r'PERIOD\s*OF\s*INSURANCE\s*[:\s\n\r]*([^\n\r]+)', text, re.I)
                if m: fields['Period of Insurance'] = m.group(1).strip()

                m = re.search(r'SUM\s*INSURED\s*[:\s\n\r]*([^\n\r]+)', text, re.I)
                if m: fields['Sum Insured'] = m.group(1).strip()

                m = re.search(r'INSURER\s*[:\s\n\r]+([A-Z\s\.,&]+(?:LTD|LIMITED|COMPANY|CO\.))', text, re.I)
                if m: fields['Insurer'] = m.group(1).strip().replace('\n', ' ')

                m = re.search(r'INSURED\s*[:\s\n\r]+([^\n\r]+)', text, re.I)
                if m and not re.search(r'^\d[\d,\s\/\-]+$', m.group(1).strip()):
                    fields['Insured Name'] = m.group(1).strip()
                else:
                    m_alt = re.search(r'M\/s\.\s*([^\n\r,]+)', text)
                    if m_alt: fields['Insured Name'] = f"M/s. {m_alt.group(1).strip()}"

                m = re.search(r'Registration\s*No\.?\s*[:\s\n\r]*([A-Z]{2}\s?[0-9]{1,2}\s?[A-Z]{1,3}\s?[0-9]{4})', text, re.I)
                if m: fields['Vehicle Registration No'] = m.group(1).strip()

                m = re.search(r'Date\s*of\s*registration\s*[:\s\n\r]*([0-3]?\d[\/-][0-1]?\d[\/-]\d{4})', text, re.I)
                if m: fields['Date of Registration'] = m.group(1).strip()

                m = re.search(r'Engine\s*No\.?\s*[:\s\n\r]*([A-Za-z0-9]+)', text, re.I)
                if m: fields['Engine Number'] = m.group(1).strip()

                m = re.search(r'Chassis\s*No\.?\s*[:\s\n\r]*([A-Za-z0-9]+)', text, re.I)
                if m: fields['Chassis Number'] = m.group(1).strip()

                m = re.search(r'Make\s*[\/\-]?\s*Model\s*[:\s\n\r]*([^\n\r]+)', text, re.I)
                if m: fields['Make / Model'] = m.group(1).strip()

                m = re.search(r'(?:Date of Loss|Date of Accident)\s*[:\s\n\r]*([0-3]?\d[\/-][0-1]?\d[\/-]\d{4})', text, re.I)
                if m: fields['Date of Loss'] = m.group(1).strip()

            # 2. DRIVING LICENSE
            elif doc_type == "Driving License":
                # Flexible DL Number extraction — not hardcoded to specific digit groups
                # 1. Label-based: match common DL label variants (DLNUMBER, DL No, Driving Licence No, etc.)
                dl_label_re = r'(?:DLNUMBER|DL[\s.\-]*(?:NUMBER|NO\.?)|DRIVING\s*LICEN[CS]E\s*(?:NO\.?|NUMBER)?|LICEN[CS]E\s*NO\.?|Details\s*Of\s*Driving\s*Licen[cs]e)'
                dl_m = re.search(dl_label_re + r'[\s:*]*([A-Z]{2}(?:[\s\-]*\d)+)', text, re.I)
                if not dl_m:
                    # 2. Standalone fallback: find 2-letter state code + digits totalling 11-16 digits
                    for _m in re.finditer(r'\b([A-Z]{2}(?:[\s\-]*\d)+)', text):
                        _candidate = re.sub(r'[\s\-]+', '', _m.group(1))
                        if re.match(r'^[A-Z]{2}\d{11,16}$', _candidate):
                            dl_m = _m
                            break
                if dl_m:
                    dl_num = re.sub(r'[\s\-]+', '', dl_m.group(1))
                    if re.match(r'^[A-Z]{2}\d{11,16}$', dl_num):
                        fields['Driving License Number'] = dl_num

                m = re.search(r'Current Status\s*[:\n\r]+\s*([A-Za-z]+)', text, re.I)
                if m: fields['Current Status'] = m.group(1).strip()

                # Holder's Name -- online (Sarathi) format
                m = re.search(r"Holder[\'\u2019]s Name\s*[:\n\r]+\s*([^\n\r]+)", text, re.I)
                if m: fields["Holder's Name"] = m.group(1).strip()
                # Physical card format: NAME <value> on same line
                if "Holder's Name" not in fields:
                    m = re.search(r'\bNAME[ \t]+([A-Z][A-Z ]+)', text)
                    if m:
                        name_val = m.group(1).strip()
                        if len(name_val) >= 3:
                            fields["Holder's Name"] = name_val

                # Father/Spouse Name -- physical card S/W/D <value>
                m = re.search(r'(?:S/W/D|S/O|D/O|W/O)[ \t]+([A-Za-z][A-Za-z ]+)', text, re.I)
                if m:
                    val = m.group(1).strip()
                    if len(val) >= 3:
                        fields["Father / Spouse Name"] = val

                m = re.search(r'Source Of Data\s*[:\n\r]+\s*([^\n\r]+)', text, re.I)
                if m: fields['Source Of Data'] = m.group(1).strip()

                # Issue Date -- online format
                m = re.search(r'Initial Issue Date\s*[:\n\r]+\s*([0-3]?\d-[A-Za-z]{3}-\d{4}|[0-3]?\d[\/\-][0-1]?\d[\/\-]\d{4})', text, re.I)
                if m: fields['Initial Issue Date'] = m.group(1).strip()
                # Physical card format: Issue Date DD-Mon-YYYY
                if 'Initial Issue Date' not in fields:
                    m = re.search(r'Issue\s*Date\s*[:\s]*(\d{1,2}[-/][A-Za-z]{3}[-/]\d{4}|\d{1,2}[-/]\d{1,2}[-/]\d{4})', text, re.I)
                    if m: fields['Issue Date'] = m.group(1).strip()

                m = re.search(r'Initial Issuing\s*\n?\s*Office\s*[:\n\r]+\s*([^\n\r]+)', text, re.I)
                if m: fields['Issuing Authority / Office'] = m.group(1).strip()
                # Physical card: Issuing Authority <value>
                if 'Issuing Authority / Office' not in fields:
                    m = re.search(r'Issuing\s*Authority\s*[:\s]*([^\n\r]+)', text, re.I)
                    if m: fields['Issuing Authority / Office'] = m.group(1).strip()

                m = re.search(r'Last Endorsed\s*\n?\s*Date\s*[:\n\r]+\s*([^\n\r]+)', text, re.I)
                if m: fields['Last Endorsed Date'] = m.group(1).strip()
                m = re.search(r'Last Endorsed\s*\n?\s*Office\s*[:\n\r]+\s*([^\n\r]+)', text, re.I)
                if m: fields['Last Endorsed Office'] = m.group(1).strip()

                m = re.search(r'Last Completed\s*\n?\s*Transaction\s*[:\n\r]+\s*([^\n\r]+)', text, re.I)
                if m: fields['Last Completed Transaction'] = m.group(1).strip()

                # Date of Birth -- online format
                m = re.search(r'Date Of Birth\s*\*?[\s\S]{0,40}?\b([0-3]?\d[-/][0-1]?\d[-/]\d{4})\b', text, re.I)
                if m: fields['Date of Birth'] = m.group(1).strip()
                # Physical card: DOB DD-Mon-YYYY or DOB DD/MM/YYYY
                if 'Date of Birth' not in fields:
                    m = re.search(r'\bDOB\s*[:\s]*(\d{1,2}[-/][A-Za-z]{3}[-/]\d{4}|\d{1,2}[-/]\d{1,2}[-/]\d{4})', text, re.I)
                    if m: fields['Date of Birth'] = m.group(1).strip()

                # Validity -- physical card Validity (NT) DD-Mon-YYYY
                m = re.search(r'Validity\s*(?:\(NT\)|\(T\))?\s*[:\s]*(\d{1,2}[-/][A-Za-z]{3}[-/]\d{4}|\d{1,2}[-/]\d{1,2}[-/]\d{4})', text, re.I)
                if m: fields['Validity'] = m.group(1).strip()

                m = re.search(r'Non-Transport\s*[\n\r\s]*From:\s*([^\n\r]+)\s*[\n\r\s]*To:\s*([^\n\r]+)', text, re.I)
                if m: fields['Non-Transport Validity'] = f"{m.group(1).strip()} to {m.group(2).strip()}"

                # Blood Group -- physical card
                m = re.search(r'Blood\s*Gr(?:ou)?p?\s*[:\s]*([ABO]+[+-]?)', text, re.I)
                if m: fields['Blood Group'] = m.group(1).strip()

                # Address -- physical card
                if 'Address' not in fields:
                    m = re.search(r'Address[ \t]*[: \t]*([^\n\r]+)', text, re.I)
                    if m: fields['Address'] = re.sub(r'\s+', ' ', m.group(1)).strip()

                covs = re.findall(r'\b(MCWG|LMV|MCWOG|TRANS|HMV|HGMV|HPMV)\b', text)
                if covs:
                    fields['Class of Vehicle (COV)'] = ', '.join(sorted(set(covs)))


            # 3. DISCHARGE VOUCHER
            elif doc_type == "Discharge Voucher":
                m = re.search(r'(?:Policy|Policy No\.?)[:\s\n\r]*([A-Za-z0-9\/\-]+)', text, re.I)
                if m: fields['Policy Number'] = m.group(1).strip()

                m = re.search(r'(?:Claim|Claim No\.?)[:\s\n\r]*([A-Za-z0-9\/\-]+)', text, re.I)
                if m: fields['Claim Number'] = m.group(1).strip()

                m = re.search(r'(?:Insured|Claimant|Favor of|Received from)[:\s\n\r]*([^\n\r]+)', text, re.I)
                if m: fields['Claimant / Insured Name'] = m.group(1).strip()

                m = re.search(r'(?:Sum of|Amount of|Net Amount|Total Amount|Settled Amount)[:\s\n\r]*(?:Rs\.?|₹)?\s*([0-9,]+(?:\.[0-9]{2})?)', text, re.I)
                if m: fields['Settlement Amount'] = f"₹{m.group(1).strip()}"

                m = re.search(r'(?:A\/c(?:\s*No\.?)?|Account\s*(?:No\.?)?)[:\s]*([0-9]{9,18})', text)
                if m: fields['Account Number'] = m.group(1).strip()

                m = re.search(r'Date\s*[:\s\n\r]*([0-3]?\d[\/-][0-1]?\d[\/-]\d{4})', text, re.I)
                if m: fields['Date'] = m.group(1).strip()

            # 4. PAYMENT RECEIPT / TAX INVOICE / TOWING BILL
            elif doc_type in ["Payment Receipt", "Tax Invoice", "Towing Bill"]:
                m = re.search(r'(?:Receipt|Invoice|Bill|Voucher)\s*No\.?\s*[:\s\n\r]*([A-Za-z0-9\/\-]+)', text, re.I)
                if m: fields['Receipt / Invoice No'] = m.group(1).strip()

                m = re.search(r'Date\s*[:\s\n\r]*([0-3]?\d[\/-][0-1]?\d[\/-]\d{4})', text, re.I)
                if m: fields['Date'] = m.group(1).strip()

                m = re.search(r'(?:Customer|Billed To|Received with thanks from)\s*[:\s\n\r]*([^\n\r]+)', text, re.I)
                if m: fields['Customer Name'] = m.group(1).strip()

                m = re.search(r'Payment\s*Mode\s*[:\s\n\r]*([A-Za-z0-9\s]+)', text, re.I)
                if m: fields['Payment Mode'] = m.group(1).strip()

                m = re.search(r'(?:Received\s*Amount|Total\s*Amount|Net\s*Amount|Invoice\s*Amount)\s*[:\s\n\r]*(?:Rs\.?|₹)?\s*([0-9,]+(?:\.[0-9]{2})?)', text, re.I)
                if m: fields['Amount'] = f"₹{m.group(1).strip()}"

                m = re.search(r'In\s*Words\s*[:\s\n\r]*([^\n\r]+)', text, re.I)
                if m: fields['Amount in Words'] = m.group(1).strip()

                m = re.search(r'Employee\s*[:\s\n\r]*([^\n\r]+)', text, re.I)
                if m: fields['Employee / Cashier'] = m.group(1).strip()

            # 5. BANK CHEQUE
            elif doc_type == "Bank Cheque":
                ifsc_match = re.search(r'\b[A-Z]{4}0[A-Z0-9]{6}\b', text)
                if ifsc_match: fields["IFSC Code"] = ifsc_match.group(0)

                acc_match = re.search(r'(?:A\/c(?:\s*No\.?)?|Account\s*(?:No\.?)?|SB\s*ACCOUNT)[:\s]*([0-9]{9,18})', text, re.I)
                if not acc_match:
                    acc_match = re.search(r'\b\d{11,16}\b', text)
                if acc_match:
                    fields["Account Number"] = acc_match.group(1) if acc_match.groups() else acc_match.group(0)

                for bank in ["State Bank of India", "Punjab National Bank", "Bank of Baroda", "HDFC Bank", "ICICI Bank", "Axis Bank", "Canara Bank", "Union Bank of India", "West Bengal Gramin Bank", "Bangiya Gramin Vikash Bank", "Kotak Mahindra Bank", "Bank of India"]:
                    if bank.upper() in t_up:
                        fields["Bank Name"] = bank
                        break

                cheque_no_match = re.search(r'["\']?([0-9]{6})["\']?\s*[:\s]*([0-9]{9})', text)
                if cheque_no_match:
                    fields["Cheque Number"] = cheque_no_match.group(1)
                    fields["MICR Code"] = cheque_no_match.group(2)
                else:
                    c_m = re.search(r'\b([0-9]{6})\b', text)
                    if c_m: fields["Cheque Number"] = c_m.group(1)

            # 6. AADHAAR CARD
            elif doc_type == "Aadhaar Card":
                aadhaar_match = re.search(r'\b\d{4}\s?\d{4}\s?\d{4}\b', text)
                if aadhaar_match: fields["Aadhaar Number"] = aadhaar_match.group(0)

                m = re.search(r'(?:Name|Applicant)[:\s]*([A-Z\s]{3,35})\b', text)
                if m: fields["Full Name"] = m.group(1).strip()

                dob_m = re.search(r'(?:DOB|Date of Birth|Birth Date)[:\s/]*([0-3]?\d[\/-][0-1]?\d[\/-]\d{4})', text, re.I)
                if dob_m: fields["Date of Birth"] = dob_m.group(1).strip()

                gender_m = re.search(r'\b(MALE|FEMALE|TRANSGENDER)\b', text, re.I)
                if gender_m: fields["Gender"] = gender_m.group(1).upper()

                co_m = re.search(r'(?:C\/O|S\/O|D\/O|W\/O)[:\s]*([^\n\r,]+)', text, re.I)
                if co_m: fields["Care of (C/O)"] = co_m.group(1).strip()

            # 7. PAN CARD
            elif doc_type == "PAN Card":
                pan_match = re.search(r'\b[A-Z]{5}[0-9]{4}[A-Z]{1}\b', text)
                if pan_match: fields["PAN Number"] = pan_match.group(0)

                dob_m = re.search(r'(?:DOB|Date of Birth|Birth Date)?[:\s/]*\b([0-3]?\d[\/-][0-1]?\d[\/-]\d{4})\b', text, re.I)
                if dob_m: fields["Date of Birth"] = dob_m.group(1).strip()

                m = re.search(r'(?:Name|Applicant)[:\s]*([A-Z\s]{3,35})\b', text)
                if m: fields["Full Name"] = m.group(1).strip()

                f_m = re.search(r'Father(?:\'s)?\s*Name[:\s]*([A-Z\s]{3,35})\b', text, re.I)
                if f_m: fields["Father's Name"] = f_m.group(1).strip()

                # If Name or Father Name were not captured by labels, detect name lines
                if "Full Name" not in fields or "Father's Name" not in fields:
                    ignore_phrases = {
                        "INCOME", "TAX", "DEPARTMENT", "GOVT", "INDIA", "PERMANENT",
                        "ACCOUNT", "NUMBER", "SIGNATURE", "CARD"
                    }
                    candidate_names = []
                    for line in text.splitlines():
                        line_clean = line.strip()
                        words = line_clean.split()
                        if len(words) >= 2 and all(w.isalpha() for w in words):
                            if not any(w.upper() in ignore_phrases for w in words):
                                candidate_names.append(line_clean)
                    if candidate_names:
                        if "Full Name" not in fields:
                            fields["Full Name"] = candidate_names[0]
                        if len(candidate_names) > 1 and "Father's Name" not in fields:
                            fields["Father's Name"] = candidate_names[1]

            # 8. SALE DEED / PROPERTY
            elif doc_type == "Sale Deed":
                m = re.search(r'(?:Deed\s*(?:No\.?|number)?|Title\s*deed\s*no\.?)[:\s\n]*([A-Za-z0-9\/\-]+)', text, re.I)
                if m: fields["Deed Number"] = m.group(1).strip()

                m = re.search(r'(?:Date\s*of\s*Registration|Registration\s*date)[:\s\n]*([0-3]?\d[\/-][0-1]?\d[\/-]\d{4}|[0-9]{1,2}\s+[A-Za-z]+\s+[0-9]{4})', text, re.I)
                if m: fields["Date of Registration"] = m.group(1).strip()

                m = re.search(r'Sub-Registr(?:y|ar)\s*(?:office|office\s*reference)?[:\s\n]*([^\n\r]+)', text, re.I)
                if m: fields["Sub-Registry Office"] = m.group(1).strip()

                m = re.search(r'(?:Purchaser|Owner|Buyer|Illustrative\s*owner)[:\s\n]*([^\n\r]+)', text, re.I)
                if m: fields["Purchaser / Buyer"] = m.group(1).strip()

                m = re.search(r'(?:Plot|Dag)\s*(?:No\.?|number)?[:\s\n]*([A-Za-z0-9\/\-, ]+)', text, re.I)
                if m: fields["Plot / Dag No"] = m.group(1).strip()

                m = re.search(r'Khatian\s*(?:No\.?|number)?[:\s\n]*([A-Za-z0-9\/\-, ]+)', text, re.I)
                if m: fields["Khatian No"] = m.group(1).strip()

                m = re.search(r'J\.?L\.?\s*(?:No\.?|number)?[:\s\n]*([0-9A-Za-z]+)', text, re.I)
                if m: fields["JL No"] = m.group(1).strip()

                m = re.search(r'Mouza\s*[:\s\n]*([A-Za-z0-9\s]+?)(?:,|\.|\n|J\.L|$)', text, re.I)
                if m: fields["Mouza"] = m.group(1).strip()

                m = re.search(r'(?:Total\s*)?Land\s*Area\s*[:\s\n]*([^\n\r]+)', text, re.I)
                if not m:
                    m = re.search(r'(?:^|\n)\s*Area[:\s\n]*([^\n\r]+)', text, re.I)
                if m: fields["Land Area"] = m.group(1).strip()

                m = re.search(r'(?:Consideration|Purchase)\s*Price\s*[:\s\n]*(?:Rs\.?|₹)?\s*([0-9,]+)', text, re.I)
                if m: fields["Consideration Price"] = f"₹{m.group(1).strip()}"

                m = re.search(r'Book\s*(?:No\.?|number)?[:\s\n]*([A-Za-z0-9]+)', text, re.I)
                if m: fields["Book No"] = m.group(1).strip()

                m = re.search(r'Volume\s*(?:No\.?|number)?[:\s\n]*([A-Za-z0-9\-]+)', text, re.I)
                if m: fields["Volume No"] = m.group(1).strip()

                m = re.search(r'Page\s*(?:No\.?|number)?[:\s\n]*([0-9\–\-]+)', text, re.I)
                if m: fields["Page No"] = m.group(1).strip()

                m = re.search(r'North[:\s\n]+([^\n\r\.]+)', text, re.I)
                if m: fields["North"] = m.group(1).strip()
                m = re.search(r'South[:\s\n]+([^\n\r\.]+)', text, re.I)
                if m: fields["South"] = m.group(1).strip()
                m = re.search(r'East[:\s\n]+([^\n\r\.]+)', text, re.I)
                if m: fields["East"] = m.group(1).strip()
                m = re.search(r'West[:\s\n]+([^\n\r\.]+)', text, re.I)
                if m: fields["West"] = m.group(1).strip()

            # 9. SALARY SLIP
            elif doc_type == "Salary Slip":
                m = re.search(r'(?:Employee\s*Name|Name)[:\s]*([A-Za-z\s]+)', text, re.I)
                if m: fields["Employee Name"] = m.group(1).strip()

                m = re.search(r'(?:Gross\s*Salary|Gross\s*Pay)[:\s]*(?:Rs\.?|₹)?\s*([0-9,]+(?:\.[0-9]{2})?)', text, re.I)
                if m: fields["Gross Salary"] = f"₹{m.group(1).strip()}"

                m = re.search(r'(?:Net\s*Salary|Net\s*Pay|Take\s*Home)[:\s]*(?:Rs\.?|₹)?\s*([0-9,]+(?:\.[0-9]{2})?)', text, re.I)
                if m: fields["Net Salary"] = f"₹{m.group(1).strip()}"

                m = re.search(r'(?:Basic\s*Pay|Basic)[:\s]*(?:Rs\.?|₹)?\s*([0-9,]+(?:\.[0-9]{2})?)', text, re.I)
                if m: fields["Basic Pay"] = f"₹{m.group(1).strip()}"

            # 10. GST CERTIFICATE
            elif doc_type == "GST Registration Certificate":
                gst = re.search(r'\b[0-9]{2}[A-Z]{5}[0-9]{4}[A-Z]{1}[1-9A-Z]{1}Z[0-9A-Z]{1}\b', text)
                if gst: fields["GSTIN"] = gst.group(0).strip()

                m = re.search(r'Legal\s*Name[:\s]*([^\n\r]+)', text, re.I)
                if m: fields["Legal Name"] = m.group(1).strip()

                m = re.search(r'Trade\s*Name[:\s]*([^\n\r]+)', text, re.I)
                if m: fields["Trade Name"] = m.group(1).strip()

                m = re.search(r'Date\s*of\s*Registration[:\s]*([0-3]?\d[\/-][0-1]?\d[\/-]\d{4})', text, re.I)
                if m: fields["Registration Date"] = m.group(1).strip()

            # 11. BORROWER IDENTITY PROFILE
            elif doc_type == "Borrower Identity Profile":
                m = re.search(r'(?:Full\s*name|Name)[:\s]*([^\n\r]+)', text, re.I)
                if m: fields["Full Name"] = m.group(1).strip()
                m = re.search(r'Title[:\s]*([^\n\r]+)', text, re.I)
                if m: fields["Title"] = m.group(1).strip()
                m = re.search(r'Father(?:\'s)?\s*Name[:\s]*([^\n\r]+)', text, re.I)
                if m: fields["Father's Name"] = m.group(1).strip()
                m = re.search(r'(?:Date\s*of\s*birth|DOB)[:\s]*([0-3]?\d[\s\/-][A-Za-z0-9]+[\s\/-]\d{4}|\d{4}-\d{2}-\d{2})', text, re.I)
                if m: fields["Date of Birth"] = m.group(1).strip()
                m = re.search(r'Gender[:\s]*([^\n\r]+)', text, re.I)
                if m: fields["Gender"] = m.group(1).strip()
                m = re.search(r'Marital\s*status[:\s]*([^\n\r]+)', text, re.I)
                if m: fields["Marital Status"] = m.group(1).strip()
                m = re.search(r'Occupation[:\s]*([^\n\r]+)', text, re.I)
                if m: fields["Occupation"] = m.group(1).strip()
                m = re.search(r'Mobile\s*number[:\s]*([0-9\s]{10,14})', text, re.I)
                if m: fields["Mobile Number"] = m.group(1).strip()
                m = re.search(r'Email[:\s]*([^\n\r\s]+@[^\n\r\s]+)', text, re.I)
                if m: fields["Email"] = m.group(1).strip()
                m = re.search(r'PAN\s*(?:reference|number|no)?[:\s]*([A-Za-z0-9]+)', text, re.I)
                if m: fields["PAN Number"] = m.group(1).strip()
                m = re.search(r'Aadhaar\s*(?:reference|number|no)?[:\s]*([X0-9\-]+)', text, re.I)
                if m: fields["Aadhaar Number"] = m.group(1).strip()
                m = re.search(r'Voter\s*ID\s*(?:reference)?[:\s]*([A-Za-z0-9\/]+)', text, re.I)
                if m: fields["Voter ID"] = m.group(1).strip()
                m = re.search(r'(?:RESIDENTIAL\s*ADDRESS|Residential\s*address)[:\s]*([^\n\r]+)', text, re.I)
                if m: fields["Residential Address"] = m.group(1).strip()

            # 12. EMPLOYMENT AND SALARY CERTIFICATE
            elif doc_type == "Employment and Salary Certificate":
                m = re.search(r'(?:^|\n)\s*Employer[:\s]*([^\n\r]+)', text, re.I)
                if m and not any(ign in m.group(1).upper() for ign in ["ISSUED", "CATEGORY", "DOCUMENT"]):
                    fields["Employer Name"] = m.group(1).strip()
                m = re.search(r'(?:Office\s*address|Employer\s*address)[:\s]*([^\n\r]+)', text, re.I)
                if m: fields["Employer Address"] = m.group(1).strip()
                m = re.search(r'(?:Employee\s*name|Name)[:\s\n]+([A-Za-z\s]{3,35})\b', text, re.I)
                if m: fields["Employee Name"] = m.group(1).strip()
                m = re.search(r'Employee\s*code[:\s]*([^\n\r]+)', text, re.I)
                if m: fields["Employee Code"] = m.group(1).strip()
                m = re.search(r'Designation[:\s]*([^\n\r]+)', text, re.I)
                if m: fields["Designation"] = m.group(1).strip()
                m = re.search(r'Department[:\s]*([^\n\r]+)', text, re.I)
                if m: fields["Department"] = m.group(1).strip()
                m = re.search(r'Employment\s*type[:\s]*([^\n\r]+)', text, re.I)
                if m: fields["Employment Type"] = m.group(1).strip()
                m = re.search(r'Date\s*of\s*joining[:\s]*([^\n\r]+)', text, re.I)
                if m: fields["Date of Joining"] = m.group(1).strip()
                m = re.search(r'Retirement\s*date(?:\s*\(illustrative\))?[:\s]*([^\n\r]+)', text, re.I)
                if m: fields["Retirement Date"] = m.group(1).strip()
                m = re.search(r'Work\s*location[:\s]*([^\n\r]+)', text, re.I)
                if m: fields["Work Location"] = m.group(1).strip()
                m = re.search(r'Basic\s*pay[:\s]*(?:INR\s*)?([0-9,.]+)', text, re.I)
                if m: fields["Basic Pay"] = m.group(1).strip()
                m = re.search(r'Dearness\s*allowance[:\s]*(?:INR\s*)?([0-9,.]+)', text, re.I)
                if m: fields["Dearness Allowance"] = m.group(1).strip()
                m = re.search(r'House\s*rent\s*allowance[:\s]*(?:INR\s*)?([0-9,.]+)', text, re.I)
                if m: fields["House Rent Allowance"] = m.group(1).strip()
                m = re.search(r'Other\s*allowance[:\s]*(?:INR\s*)?([0-9,.]+)', text, re.I)
                if m: fields["Other Allowance"] = m.group(1).strip()
                m = re.search(r'Gross\s*monthly\s*salary[:\s]*(?:INR\s*)?([0-9,.]+)', text, re.I)
                if m: fields["Gross Monthly Salary"] = m.group(1).strip()
                m = re.search(r'Provident\s*fund\s*deduction[:\s]*(?:INR\s*)?([0-9,.]+)', text, re.I)
                if m: fields["Provident Fund Deduction"] = m.group(1).strip()
                m = re.search(r'Professional\s*tax[:\s]*(?:INR\s*)?([0-9,.]+)', text, re.I)
                if m: fields["Professional Tax"] = m.group(1).strip()
                m = re.search(r'Other\s*deductions[:\s]*(?:INR\s*)?([0-9,.]+)', text, re.I)
                if m: fields["Other Deductions"] = m.group(1).strip()
                m = re.search(r'Total\s*deductions[:\s]*(?:INR\s*)?([0-9,.]+)', text, re.I)
                if m: fields["Total Deductions"] = m.group(1).strip()
                m = re.search(r'Net\s*(?:take-home\s*)?salary[:\s]*(?:INR\s*)?([0-9,.]+)', text, re.I)
                if m: fields["Net Salary"] = m.group(1).strip()
                m = re.search(r'Certificate\s*date[:\s]*([^\n\r]+)', text, re.I)
                if m: fields["Certificate Date"] = m.group(1).strip()

            # 13. BANK STATEMENT / SALARY ACCOUNT STATEMENT
            elif doc_type == "Bank Statement":
                m = re.search(r'(?:^|\n)\s*Bank\s*name[:\s]*([^\n\r]+)', text, re.I)
                if not m:
                    m = re.search(r'(?:^|\n)\s*Bank[:\s]*([^\n\r]+)', text, re.I)
                if m: fields["Bank Name"] = m.group(1).strip()
                m = re.search(r'Branch[:\s]*([^\n\r]+)', text, re.I)
                if m: fields["Branch"] = m.group(1).strip()
                m = re.search(r'Account\s*holder[:\s]*([^\n\r]+)', text, re.I)
                if m: fields["Account Holder"] = m.group(1).strip()
                m = re.search(r'Account\s*type[:\s]*([^\n\r]+)', text, re.I)
                if m: fields["Account Type"] = m.group(1).strip()
                m = re.search(r'Account\s*number[:\s]*([0-9A-Za-z\-]+)', text, re.I)
                if m: fields["Account Number"] = m.group(1).strip()
                m = re.search(r'IFSC(?:\s*code)?[:\s]*([A-Za-z0-9]+)', text, re.I)
                if m: fields["IFSC"] = m.group(1).strip()
                m = re.search(r'Statement\s*period[:\s]*([^\n\r]+)', text, re.I)
                if m: fields["Statement Period"] = m.group(1).strip()
                m = re.search(r'Closing\s*balance(?:\s*shown)?[:\s]*(?:INR\s*)?([0-9,.]+)', text, re.I)
                if m: fields["Closing Balance"] = m.group(1).strip()

            # 14. LOAN APPLICATION / REQUEST LETTER
            elif doc_type == "Loan Application":
                m = re.search(r'(?:Applicant\s*/\s*borrower|Applicant\s*name|Full\s*name)[:\s\n]+([A-Za-z ]{3,35})', text, re.I)
                if not m:
                    m = re.search(r'\bI,\s*([A-Za-z ]{3,35})\b,?\s*son of', text, re.I)
                if m and not any(ign in m.group(1).upper() for ign in ["LETTER", "DOCUMENT", "CATEGORY", "APPLICATION"]):
                    fields["Applicant Name"] = m.group(1).strip()

                m = re.search(r'(?:son\s*of|Father(?:\'s)?\s*Name)[:\s\n]+([A-Za-z ]{3,35})', text, re.I)
                if m: fields["Father's Name"] = m.group(1).strip()

                m = re.search(r'(?:^|\n)\s*([A-Za-z ]+Branch)', text, re.I)
                if m and not any(ign in m.group(1).upper() for ign in ["THE", "MANAGER", "BANK", "DOCUMENT"]):
                    fields["Branch"] = m.group(1).strip()

                m = re.search(r'(?:Loan\s*purpose|Purpose)[:\s\n]*([^\n\r]+)', text, re.I)
                if m: fields["Purpose"] = m.group(1).strip()

                m = re.search(r'under\s*the\s*([A-Za-z\s]+Scheme)', text, re.I)
                if not m:
                    m = re.search(r'Scheme[:\s\n]*([^\n\r]+)', text, re.I)
                if m: fields["Scheme"] = m.group(1).strip()

                m = re.search(r'(?:Amount\s*requested|Requested\s*(?:loan\s*)?amount)[:\s\n]*(?:INR\s*)?([0-9,.]+)', text, re.I)
                if m: fields["Requested Loan Amount"] = m.group(1).strip()

                m = re.search(r'(?:Proposed\s*repayment\s*period|Repayment\s*tenure|Term\s*months)[:\s\n]*([^\n\r]+)', text, re.I)
                if m: fields["Term Months"] = m.group(1).strip()

                m = re.search(r'Salary\s*account\s*(?:number|reference)?[:\s\n]*([A-Za-z0-9]+)', text, re.I)
                if m: fields["Salary Account Reference"] = m.group(1).strip()

                m = re.search(r'(?:Proposed\s*guarantor|Guarantor)[:\s\n]*([A-Za-z ]{3,35})', text, re.I)
                if m: fields["Guarantor Full Name"] = m.group(1).strip()

                m = re.search(r'(?:Proposed\s*property|Property\s*location)[:\s\n]*([^\n\r]+)', text, re.I)
                if m: fields["Proposed Property"] = m.group(1).strip()

                m = re.search(r'Application\s*date[:\s\n]*([^\n\r]+)', text, re.I)
                if m: fields["Application Date"] = m.group(1).strip()

            # 15. GUARANTOR IDENTITY PROFILE
            elif doc_type == "Guarantor Profile":
                m = re.search(r'Full\s*name[:\s]*([^\n\r]+)', text, re.I)
                if m: fields["Guarantor Full Name"] = m.group(1).strip()
                m = re.search(r'Title[:\s]*([^\n\r]+)', text, re.I)
                if m: fields["Title"] = m.group(1).strip()
                m = re.search(r'Father(?:\'s)?\s*Name[:\s]*([^\n\r]+)', text, re.I)
                if m: fields["Guarantor Father Name"] = m.group(1).strip()
                m = re.search(r'Date\s*of\s*birth[:\s]*([^\n\r]+)', text, re.I)
                if m: fields["Guarantor DOB"] = m.group(1).strip()
                m = re.search(r'Gender[:\s]*([^\n\r]+)', text, re.I)
                if m: fields["Guarantor Gender"] = m.group(1).strip()
                m = re.search(r'Marital\s*status[:\s]*([^\n\r]+)', text, re.I)
                if m: fields["Guarantor Marital Status"] = m.group(1).strip()
                m = re.search(r'Relationship(?:\s*to\s*applicant)?[:\s]*([^\n\r]+)', text, re.I)
                if m: fields["Relationship"] = m.group(1).strip()
                m = re.search(r'Occupation[:\s]*([^\n\r]+)', text, re.I)
                if m: fields["Guarantor Occupation"] = m.group(1).strip()
                m = re.search(r'Mobile\s*number[:\s]*([0-9\s]{10,14})', text, re.I)
                if m: fields["Guarantor Mobile Number"] = m.group(1).strip()
                m = re.search(r'PAN\s*(?:reference|no)?[:\s]*([A-Za-z0-9]+)', text, re.I)
                if m: fields["Guarantor PAN Number"] = m.group(1).strip()
                m = re.search(r'Aadhaar\s*(?:reference|no)?[:\s]*([X0-9\-]+)', text, re.I)
                if m: fields["Guarantor Aadhaar"] = m.group(1).strip()
                m = re.search(r'(?:RESIDENTIAL\s*ADDRESS|Residential\s*address)[:\s]*([^\n\r]+)', text, re.I)
                if m: fields["Residential Address"] = m.group(1).strip()

            # 16. GUARANTOR INCOME AND NET-WORTH
            elif doc_type == "Guarantor Income and Net-Worth":
                m = re.search(r'(?:Guarantor\s*name|Name)[:\s]*([A-Za-z ]{3,35})', text, re.I)
                if m and not any(ign in m.group(1).upper() for ign in ["DECLARATION", "STATEMENT", "FINANCIAL"]):
                    fields["Guarantor Full Name"] = m.group(1).strip()
                m = re.search(r'Father(?:\'s)?\s*Name[:\s]*([^\n\r]+)', text, re.I)
                if m: fields["Guarantor Father Name"] = m.group(1).strip()
                m = re.search(r'Employer[:\s]*([^\n\r]+)', text, re.I)
                if m: fields["Guarantor Employer"] = m.group(1).strip()
                m = re.search(r'Designation[:\s]*([^\n\r]+)', text, re.I)
                if m: fields["Guarantor Designation"] = m.group(1).strip()
                m = re.search(r'Monthly\s*gross\s*(?:salary|income)[:\s\n]*(?:INR\s*)?([0-9,.]+)', text, re.I)
                if m: fields["Guarantor Gross Salary"] = m.group(1).strip()
                m = re.search(r'Monthly\s*net\s*(?:salary|income)[:\s\n]*(?:INR\s*)?([0-9,.]+)', text, re.I)
                if m: fields["Guarantor Net Salary"] = m.group(1).strip()
                m = re.search(r'Immovable\s*property(?:.*?market\s*value)?[:\s]*(?:INR\s*)?([0-9,.]+)', text, re.I)
                if m: fields["Guarantor Immovable Property"] = m.group(1).strip()
                m = re.search(r'Movable\s*assets[:\s]*(?:INR\s*)?([0-9,.]+)', text, re.I)
                if m: fields["Guarantor Movable Assets"] = m.group(1).strip()
                m = re.search(r'Total\s*assets[:\s]*(?:INR\s*)?([0-9,.]+)', text, re.I)
                if m: fields["Guarantor Total Assets"] = m.group(1).strip()
                m = re.search(r'Liabilities(?:.*?existing\s*loans)?[:\s]*(?:INR\s*)?([0-9,.]+)', text, re.I)
                if m: fields["Guarantor Total Liabilities"] = m.group(1).strip()
                m = re.search(r'Net\s*worth[:\s]*(?:INR\s*)?([0-9,.]+)', text, re.I)
                if m: fields["Guarantor Net Worth"] = m.group(1).strip()
                m = re.search(r'Dependants[:\s]*([^\n\r]+)', text, re.I)
                if m: fields["Dependants"] = m.group(1).strip()

            # 17. PROPERTY TAX RECEIPT
            elif doc_type == "Property Tax Receipt":
                m = re.search(r'Receipt\s*reference[:\s]*([^\n\r]+)', text, re.I)
                if m: fields["Receipt Reference"] = m.group(1).strip()
                m = re.search(r'Issuing\s*authority[:\s]*([^\n\r]+)', text, re.I)
                if m: fields["Issuing Authority"] = m.group(1).strip()
                m = re.search(r'Assessee\s*\/\s*Owner[:\s]*([^\n\r,]+)', text, re.I)
                if m: fields["Owner"] = m.group(1).strip()
                m = re.search(r'Holding\s*\/\s*Assessment\s*number[:\s]*([^\n\r,]+)', text, re.I)
                if m: fields["Holding / Assessment Number"] = m.group(1).strip()
                m = re.search(r'Tax\s*period[:\s]*([^\n\r]+)', text, re.I)
                if m: fields["Tax Period"] = m.group(1).strip()
                m = re.search(r'Amount\s*paid[:\s]*(?:INR\s*)?([0-9,.]+)', text, re.I)
                if m: fields["Amount Paid"] = m.group(1).strip()
                m = re.search(r'Payment\s*date[:\s]*([^\n\r]+)', text, re.I)
                if m: fields["Payment Date"] = m.group(1).strip()
                m = re.search(r'(?:^|\n)\s*Plot\s*(?:No\.?|number)?[:\s\n]*([A-Za-z0-9\/\-]+)', text, re.I)
                if m: fields["Plot No"] = m.group(1).strip()

            # 18. PROPERTY VALUATION REPORT
            elif doc_type == "Property Valuation Report":
                m = re.search(r'Report\s*reference[:\s]*([^\n\r]+)', text, re.I)
                if m: fields["Report Reference"] = m.group(1).strip()
                m = re.search(r'Inspection\s*date[:\s]*([^\n\r]+)', text, re.I)
                if m: fields["Inspection Date"] = m.group(1).strip()
                m = re.search(r'Applicant\s*\/\s*Proposed\s*owner[:\s]*([^\n\r,]+)', text, re.I)
                if m: fields["Applicant Name"] = m.group(1).strip()
                m = re.search(r'Property\s*location[:\s]*([^\n\r]+)', text, re.I)
                if m: fields["Property Location"] = m.group(1).strip()
                m = re.search(r'Mouza(?:\s*\/\s*J\.?L\.?)?[:\s\n]*([A-Za-z0-9\s]+?)(?:\/|\n|$)', text, re.I)
                if m: fields["Mouza"] = m.group(1).strip()
                m = re.search(r'J\.?L\.?(?:\s*No\.?|\s*Number)?[:\s\n]*([0-9A-Za-z]+)', text, re.I)
                if m: fields["JL No"] = m.group(1).strip()
                m = re.search(r'(?:^|\n)\s*Plot\s*(?:No\.?|number)?[:\s\n]*([A-Za-z0-9\/\-]+)', text, re.I)
                if m: fields["Plot No"] = m.group(1).strip()
                m = re.search(r'Khatian\s*(?:No\.?|number)?[:\s\n]*([^\n\r,]+)', text, re.I)
                if m: fields["Khatian No"] = m.group(1).strip()
                m = re.search(r'Land\s*area[:\s\n]*([^\n\r,]+)', text, re.I)
                if m: fields["Land Area"] = m.group(1).strip()
                m = re.search(r'Built-up\s*area\s*\/\s*Proposed[:\s\n]*([^\n\r,]+)', text, re.I)
                if m: fields["Proposed Buildup Area"] = m.group(1).strip()
                m = re.search(r'Estimated\s*land\s*value[:\s\n]*(?:INR\s*)?([0-9,.]+)', text, re.I)
                if m: fields["Estimated Land Value"] = m.group(1).strip()
                m = re.search(r'Estimated\s*construction\s*cost(?:\s*\/\s*Proposed)?[:\s\n]*(?:INR\s*)?([0-9,.]+)', text, re.I)
                if m: fields["Estimated Construction Cost"] = m.group(1).strip()
                m = re.search(r'(?:Estimated\s*total\s*market\s*value|Total\s*estimated\s*property\s*value)[:\s\n]*(?:INR\s*)?([0-9,.]+)', text, re.I)
                if m: fields["Total Estimated Property Value"] = m.group(1).strip()
                m = re.search(r'Estimated\s*realisable\s*value[:\s\n]*(?:INR\s*)?([0-9,.]+)', text, re.I)
                if m: fields["Estimated Realisable Value"] = m.group(1).strip()
                m = re.search(r'Estimated\s*distress\s*value[:\s\n]*(?:INR\s*)?([0-9,.]+)', text, re.I)
                if m: fields["Estimated Distress Value"] = m.group(1).strip()
                m = re.search(r'North[:\s]*([^\.\n]+)', text, re.I)
                if m: fields["North"] = m.group(1).strip()
                m = re.search(r'South[:\s]*([^\.\n]+)', text, re.I)
                if m: fields["South"] = m.group(1).strip()
                m = re.search(r'East[:\s]*([^\.\n]+)', text, re.I)
                if m: fields["East"] = m.group(1).strip()
                m = re.search(r'West[:\s]*([^\.\n]+)', text, re.I)
                if m: fields["West"] = m.group(1).strip()

            # 19. LEGAL SEARCH REPORT
            elif doc_type == "Legal Search Report":
                m = re.search(r'Report\s*reference[:\s]*([^\n\r]+)', text, re.I)
                if m: fields["Report Reference"] = m.group(1).strip()
                m = re.search(r'Proposed\s*borrower\s*\/\s*owner[:\s]*([^\n\r,]+)', text, re.I)
                if m: fields["Owner"] = m.group(1).strip()
                m = re.search(r'Search\s*period[:\s]*([^\n\r]+)', text, re.I)
                if m: fields["Search Period"] = m.group(1).strip()
                m = re.search(r'Sub-Registry\s*office[:\s]*([^\n\r]+)', text, re.I)
                if m: fields["Sub-Registry Office"] = m.group(1).strip()
                m = re.search(r'Deed\s*reference\s*searched[:\s]*([^\n\r]+)', text, re.I)
                if m: fields["Deed Reference Searched"] = m.group(1).strip()
                m = re.search(r'Book[:\s]*([^\n\r]+)', text, re.I)
                if m: fields["Book No"] = m.group(1).strip()
                m = re.search(r'Volume[:\s]*([^\n\r]+)', text, re.I)
                if m: fields["Volume No"] = m.group(1).strip()
                m = re.search(r'Page[:\s]*([^\n\r]+)', text, re.I)
                if m: fields["Page No"] = m.group(1).strip()
                m = re.search(r'Encumbrance\s*result[:\s]*([^\n\r]+)', text, re.I)
                if m: fields["Encumbrance Result"] = m.group(1).strip()

            # 20. LOAN SANCTION ADVICE
            elif doc_type == "Loan Sanction Advice":
                m = re.search(r'Borrower[:\s]*([^\n\r]+)', text, re.I)
                if m: fields["Full Name"] = m.group(1).strip()
                m = re.search(r'Loan\s*account\s*number[:\s]*([A-Za-z0-9\-]+)', text, re.I)
                if m: fields["Loan Account Number"] = m.group(1).strip()
                m = re.search(r'Branch[:\s]*([^\n\r]+)', text, re.I)
                if m: fields["Branch"] = m.group(1).strip()
                m = re.search(r'Scheme[:\s]*([^\n\r]+)', text, re.I)
                if m: fields["Scheme"] = m.group(1).strip()
                m = re.search(r'Purpose[:\s]*([^\n\r]+)', text, re.I)
                if m: fields["Purpose"] = m.group(1).strip()
                m = re.search(r'Requested\s*amount[:\s]*(?:INR\s*)?([0-9,.]+)', text, re.I)
                if m: fields["Requested Loan Amount"] = m.group(1).strip()
                m = re.search(r'(?:Illustrative\s*)?sanctioned\s*amount[:\s]*(?:INR\s*)?([0-9,.]+)', text, re.I)
                if m: fields["Sanctioned Amount"] = m.group(1).strip()
                m = re.search(r'Interest\s*rate[:\s]*([^\n\r]+)', text, re.I)
                if m: fields["Rate Of Interest"] = m.group(1).strip()
                m = re.search(r'Repayment\s*period[:\s]*([^\n\r]+)', text, re.I)
                if m: fields["Term Months"] = m.group(1).strip()
                m = re.search(r'(?:Illustrative\s*)?EMI[:\s]*(?:INR\s*)?([0-9,.]+)', text, re.I)
                if m: fields["EMI"] = m.group(1).strip()
                m = re.search(r'(?:Sample\s*)?sanction\s*date[:\s]*([^\n\r]+)', text, re.I)
                if m: fields["Sanction Date"] = m.group(1).strip()
                m = re.search(r'Proposed\s*security[:\s]*([^\n\r]+)', text, re.I)
                if m: fields["Proposed Security"] = m.group(1).strip()
                m = re.search(r'Guarantor[:\s]*([^\n\r]+)', text, re.I)
                if m: fields["Guarantor Full Name"] = m.group(1).strip()

            # ---------------------------------------------------------
            # COMMON PATTERNS PASS (Dates, Emails, Mobile, GSTIN, PIN)
            # ---------------------------------------------------------
            if "Email" not in fields:
                em = re.search(r'[\w.+-]+@[\w-]+\.[\w.-]+', text)
                if em: fields["Email"] = em.group(0).strip()

            if "Mobile Number" not in fields and "Phone Number" not in fields:
                mob = re.search(r'(?:Mob(?:ile)?|Phone|Tel)?[:\s]*\b([6-9]\d{9})\b', text, re.I)
                if mob: fields["Mobile Number"] = mob.group(1).strip()

            if "GSTIN" not in fields:
                gst = re.search(r'\b[0-9]{2}[A-Z]{5}[0-9]{4}[A-Z]{1}[1-9A-Z]{1}Z[0-9A-Z]{1}\b', text)
                if gst: fields["GSTIN"] = gst.group(0).strip()

            if "PIN Code" not in fields and "Bank Cheque" not in doc_type:
                pin = re.search(r'(?:PIN(?:\s*Code)?[:\s]*)(\b[1-9][0-9]{5}\b)', text, re.I)
                if pin: fields["PIN Code"] = pin.group(1).strip()

            if "Amount" not in fields and doc_type in ["Official Document", "Affidavit", "Bank Statement"]:
                amount_matches = re.findall(r'(?:Rs\.?|₹|INR)\s*([\d,]+(?:\.\d{2})?)', text)
                if amount_matches:
                    fields["Amount"] = f"₹{amount_matches[0]}"

            # ---------------------------------------------------------
            # GENERIC HEURISTIC KEY-VALUE PARSER
            # ---------------------------------------------------------
            DISALLOWED_KEYS = {
                'http', 'https', 'pur_cd', 'www', 'note', 'url', 'from', 'to',
                'terms of uses', 'disclaimer', 'page', 'signature', 'thumb',
                'reset', 'check status', 'details of driving license',
                'holder signature', 'issue auth signature', 'aadhaar no',
                'authorisation to drive', 'date of issue', 'form',
                'name', 'dlnumber', 'dl number', 'inv carr no', 'inv carr no.'
            }
            existing_keys_lower = {k.lower(): k for k in fields.keys()}

            lines = [l.strip() for l in text.split('\n') if l.strip()]

            # Pass A: Single line Key: Value
            for line in lines:
                if ":" in line:
                    parts = line.split(":", 1)
                    k_candidate = parts[0].strip()
                    v_candidate = parts[1].strip()

                    k_clean = re.sub(r'^[0-9\.\-\*\)\(\]\[a-zA-Z]\s*[\.\)\]]', '', k_candidate).strip()
                    k_clean = re.sub(r'^[^\w]+|[^\w]+$', '', k_clean).strip()
                    v_clean = re.sub(r'^[^\w]+|[^\w]+$', '', v_candidate).strip()

                    if re.match(r'^[A-Za-z]', k_clean) and 2 <= len(k_clean) <= 32 and 2 <= len(v_clean) <= 80:
                        if not v_candidate.lower().startswith(('http', '//', 'www')) and not k_clean.lower().startswith(('http', '//', 'www', '(')):
                            norm_k = k_clean.replace('_', ' ').title()
                            if norm_k.lower() not in DISALLOWED_KEYS and norm_k.lower() not in existing_keys_lower:
                                if v_clean.lower() not in ['na', 'null', 'none', '-', '--', 'nil', 'unknown', 'n/a', '']:
                                    fields[norm_k] = v_clean
                                    existing_keys_lower[norm_k.lower()] = norm_k

            # Pass B: Consecutive lines Label -> Value (common in OCR tables / forms)
            # Skip for well-classified document types that have dedicated extractors
            SKIP_PASS_B_TYPES = {"Driving License", "PAN Card", "Aadhaar Card", "Bank Cheque", "Motor Survey Report"}
            if doc_type not in SKIP_PASS_B_TYPES:
                LABEL_SUFFIXES = ("name", "no", "number", "date", "status", "office", "code", "branch", "bank", "address", "class", "validity", "category", "type", "authority", "officer", "transaction")
                for i in range(len(lines) - 1):
                    cur = lines[i].strip()
                    nxt = lines[i+1].strip()
                    cur_clean = re.sub(r'[:\*]+$', '', cur).strip()
                    if 3 <= len(cur_clean) <= 35 and re.match(r'^[A-Za-z]', cur_clean):
                        if any(cur_clean.lower().endswith(sfx) or sfx in cur_clean.lower().split() for sfx in LABEL_SUFFIXES):
                            if 1 <= len(nxt) <= 80 and not nxt.lower().startswith(('http', '//', 'www')):
                                norm_k = cur_clean.replace('_', ' ').title()
                                if norm_k.lower() not in DISALLOWED_KEYS and norm_k.lower() not in existing_keys_lower:
                                    if nxt.lower() not in ['na', 'null', 'none', '-', '--', 'nil', 'unknown', 'n/a', '']:
                                        fields[norm_k] = nxt
                                        existing_keys_lower[norm_k.lower()] = norm_k

            summary = self.generate_summary(doc_type, fields)
            result_docs.append({
                "document_name": doc_name,
                "parent_file": doc.get("parent_file", doc_name),
                "page_number": doc.get("page_number", 1),
                "document_type": doc_type,
                "fields": fields,
                "summary": summary,
                "ocr_text": text
            })

        return {
            "engine_used": "Local Rule & Heuristic Engine (Offline)",
            "documents": result_docs
        }

if __name__ == "__main__":
    extractor = AIExtractor()
    sample = [{
        "id": 1,
        "name": "test_cheque.jpg",
        "ocr_text": "STATE BANK OF INDIA\nFARIDKOT PUNJAB 151203\nIFS CODE: SBIN0050660\nA/c No: 18234454256\nSB ACCOUNT"
    }]
    res = extractor.extract_from_documents(sample)
    print("AI Extractor Test Output:\n", json.dumps(res, indent=2))
