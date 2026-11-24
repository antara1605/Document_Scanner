import os
import uuid
import traceback
from datetime import datetime
from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
from dotenv import load_dotenv

from ocr_service import OCRService
from ai_extractor import AIExtractor
from consolidator import MultiDocumentConsolidator
from excel_service import populate_excel

load_dotenv(os.path.join(os.path.abspath(os.path.dirname(__file__)), ".env"))

app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'dscan-extract-2026')
app.config['TEMPLATES_AUTO_RELOAD'] = True
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

UPLOAD_FOLDER = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'uploads')
OUTPUT_FOLDER = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'output')
ALLOWED_EXTENSIONS = {'pdf', 'png', 'jpg', 'jpeg', 'bmp', 'tiff', 'webp'}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

def get_template_path():
    local_copy = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'template_copy', 'WBGB_HBL_LOAN.xlsm')
    if os.path.exists(local_copy):
        return local_copy
    desktop = os.path.join(os.environ.get('USERPROFILE', ''), 'Desktop', 'WBGB HBL LOAN (2).xlsm')
    if os.path.exists(desktop):
        return desktop
    return local_copy

ocr_service = OCRService()
ai_extractor = AIExtractor()

@app.after_request
def add_header(response):
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload-and-process', methods=['POST'])
def upload_and_process():
    files = request.files.getlist('documents')
    if not files or len(files) == 0 or all(f.filename == '' for f in files):
        files = request.files.getlist('document')

    valid_files = [f for f in files if f and f.filename.strip() and allowed_file(f.filename)]
    if not valid_files:
        return jsonify({
            "success": False,
            "error": "No valid documents uploaded. Please upload JPG, PNG, or PDF files."
        }), 400

    try:
        # Step 1: Run OCR on all uploaded documents
        doc_payload = []
        processed_files = []
        ocr_results_map = {}

        print(f"[{datetime.now().strftime('%H:%M:%S')}] Received batch of {len(valid_files)} document(s).", flush=True)
        for idx, file in enumerate(valid_files):
            raw_filename = file.filename
            orig_name = secure_filename(raw_filename)
            ext = raw_filename.rsplit('.', 1)[-1].lower() if '.' in raw_filename else 'bin'
            if not orig_name:
                orig_name = f"document_{idx+1}.{ext}"

            unique_name = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}_{orig_name}"
            save_path = os.path.join(UPLOAD_FOLDER, unique_name)
            file.save(save_path)

            print(f"  [{idx+1}/{len(valid_files)}] Running OCR on: {orig_name} ...", flush=True)
            try:
                ocr_result = ocr_service.process_file(save_path)
            finally:
                if os.path.exists(save_path):
                    try:
                        os.remove(save_path)
                    except Exception as e:
                        print(f"Warning: Failed to delete temp file {save_path}: {e}")
            full_text = ocr_result.get('full_text', '')
            pages = ocr_result.get('pages', [])

            doc_payload.append({
                "id": idx + 1,
                "name": orig_name,
                "category": "Uploaded Document",
                "ocr_text": full_text,
                "pages": pages
            })
            processed_files.append(orig_name)
            ocr_results_map[orig_name] = ocr_result

        # Step 2: Run AI Extraction across all documents together
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Analyzing all {len(doc_payload)} document(s) with AI...", flush=True)
        extraction_result = ai_extractor.extract_from_documents(doc_payload)
        extracted_docs = extraction_result.get("documents", [])
        engine_used = extraction_result.get("engine_used", "Local Rule & Heuristic Engine (Offline)")

        # Step 3: Package final results with clean string fields and actual OCR text
        final_doc_results = []
        target_doc_list = extracted_docs if extracted_docs else [{"document_name": p, "fields": {}} for p in processed_files]

        for idx, doc in enumerate(target_doc_list):
            p_name = doc.get("document_name") or (processed_files[min(idx, len(processed_files)-1)] if processed_files else "Document")
            parent_file = doc.get("parent_file", processed_files[0] if processed_files else p_name)
            p_num = doc.get("page_number", 1)
            ocr_info = ocr_results_map.get(parent_file, {})

            raw_fields = doc.get("fields", {})
            doc_type = doc.get("document_type", "Document")

            clean_fields = {}
            SKIP_VALUES = {'', 'none', 'null', 'nil', 'n/a', 'na', '-', '--', '---', 'not applicable', 'not specified', 'unknown'}
            if isinstance(raw_fields, dict):
                for k, v in raw_fields.items():
                    val_str = ""
                    if isinstance(v, dict):
                        val = v.get("value", v)
                        val_str = str(val).strip() if val is not None else ""
                    elif isinstance(v, list):
                        val_str = ", ".join(str(i).strip() for i in v if str(i).strip())
                    elif v is not None:
                        val_str = str(v).strip()

                    if val_str and val_str.lower() not in SKIP_VALUES:
                        clean_fields[str(k).strip()] = val_str
            elif isinstance(raw_fields, list):
                for item in raw_fields:
                    if isinstance(item, dict):
                        for k, v in item.items():
                            val_str = str(v).strip() if v is not None else ""
                            if val_str and val_str.lower() not in SKIP_VALUES:
                                clean_fields[str(k).strip()] = val_str

            doc_summary = doc.get("summary") or ai_extractor.generate_summary(doc_type, clean_fields)
            doc_ocr_text = doc.get("ocr_text") or ocr_info.get("full_text", "")

            final_doc_results.append({
                "document_name": p_name,
                "parent_file": parent_file,
                "page_number": p_num,
                "document_type": doc_type,
                "summary": doc_summary,
                "fields": clean_fields,
                "ocr_text": doc_ocr_text,
                "page_count": 1,
                "confidence": ocr_info.get("average_confidence", 0.90)
            })

        total_fields = sum(len(d["fields"]) for d in final_doc_results)
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Extraction complete. {total_fields} fields found across {len(final_doc_results)} document(s).", flush=True)

        # Step 4: Run Multi-Document Consolidation & Conflict Handling
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Consolidating and resolving roles across {len(final_doc_results)} document(s)...", flush=True)
        consolidator = MultiDocumentConsolidator()
        norm_data, audit_trail, cons_summary = consolidator.consolidate({"documents": final_doc_results, "engine_used": engine_used})
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Consolidation complete. {cons_summary['verified_multi_source_fields']} verified across docs, {cons_summary['conflicting_fields']} conflicts needing review.", flush=True)

        return jsonify({
            "success": True,
            "processed_files_count": len(processed_files),
            "processed_files": processed_files,
            "documents": final_doc_results,
            "total_fields": total_fields,
            "engine_used": engine_used,
            "consolidated_data": norm_data.to_dict(),
            "audit_trail": [entry.to_dict() for entry in audit_trail],
            "consolidation_summary": cons_summary
        })

    except Exception as e:
        traceback.print_exc()
        error_msg = str(e)
        return jsonify({
            "success": False,
            "error": f"Extraction failed: {error_msg}"
        }), 500

@app.route('/generate-excel', methods=['POST'])
def generate_excel():
    data = request.get_json() or {}
    approved_data = data.get('approved_data')
    manual_edits = data.get('manual_edits', [])

    if not approved_data:
        return jsonify({"success": False, "error": "No approved data provided"}), 400

    template_path = get_template_path()
    if not os.path.exists(template_path):
        return jsonify({"success": False, "error": f"Template workbook not found at {template_path}"}), 404

    # Extract borrower name for clean output filename
    borrower_name = "Loan_Applicant"
    try:
        b_val = approved_data.get("borrower", {}).get("full_name")
        if isinstance(b_val, dict):
            b_name = b_val.get("value")
        else:
            b_name = b_val
        if b_name and str(b_name).strip():
            borrower_name = secure_filename(str(b_name).strip().replace(" ", "_"))
    except Exception:
        pass

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_filename = f"WBGB_HBL_LOAN_{borrower_name}_{timestamp}.xlsm"
    output_path = os.path.join(OUTPUT_FOLDER, out_filename)

    try:
        report = populate_excel(
            normalized_data=approved_data,
            template_path=template_path,
            output_path=output_path,
            overwrite_existing_output=True
        )

        if report.get("success"):
            report["manual_edits_count"] = len(manual_edits)
            report["manual_edits"] = manual_edits
            return jsonify({
                "success": True,
                "filename": out_filename,
                "download_url": f"/download-excel/{out_filename}",
                "report": report
            })
        else:
            return jsonify({
                "success": False,
                "error": report.get("error", "Failed to generate Excel file"),
                "report": report
            }), 500
    except Exception as e:
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/download-excel/<path:filename>')
def download_excel(filename):
    safe_filename = secure_filename(filename)
    file_path = os.path.join(OUTPUT_FOLDER, safe_filename)
    if not os.path.exists(file_path):
        return jsonify({"error": "File not found"}), 404
    return send_from_directory(OUTPUT_FOLDER, safe_filename, as_attachment=True)


if __name__ == '__main__':
    print("=" * 60)
    print("  Document Scanner - Extract & Display")
    print("  Server running on http://127.0.0.1:5000")
    print("=" * 60)
    app.run(host='127.0.0.1', port=5000, debug=False)
