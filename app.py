import os
import uuid
import traceback
from datetime import datetime
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from dotenv import load_dotenv

from ocr_service import OCRService
from ai_extractor import AIExtractor

load_dotenv(os.path.join(os.path.abspath(os.path.dirname(__file__)), ".env"))

app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'dscan-extract-2026')
app.config['TEMPLATES_AUTO_RELOAD'] = True
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

UPLOAD_FOLDER = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'uploads')
ALLOWED_EXTENSIONS = {'pdf', 'png', 'jpg', 'jpeg', 'bmp', 'tiff', 'webp'}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

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
        for idx, p_name in enumerate(processed_files):
            ocr_info = ocr_results_map.get(p_name, {})
            # Match document from AI result
            matching_ai_doc = None
            for d in extracted_docs:
                if d.get("document_name") == p_name:
                    matching_ai_doc = d
                    break
            if not matching_ai_doc and idx < len(extracted_docs):
                matching_ai_doc = extracted_docs[idx]

            raw_fields = matching_ai_doc.get("fields", {}) if matching_ai_doc else {}
            doc_type = matching_ai_doc.get("document_type", "Document") if matching_ai_doc else "Document"

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

            doc_summary = ""
            if matching_ai_doc and matching_ai_doc.get("summary"):
                doc_summary = matching_ai_doc.get("summary")
            else:
                doc_summary = ai_extractor.generate_summary(doc_type, clean_fields)

            final_doc_results.append({
                "document_name": p_name,
                "document_type": doc_type,
                "summary": doc_summary,
                "fields": clean_fields,
                "ocr_text": ocr_info.get("full_text", ""),
                "page_count": ocr_info.get("page_count", 1),
                "confidence": ocr_info.get("average_confidence", 0.85)
            })

        total_fields = sum(len(d["fields"]) for d in final_doc_results)
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Extraction complete. {total_fields} fields found across {len(final_doc_results)} document(s).", flush=True)

        return jsonify({
            "success": True,
            "processed_files_count": len(processed_files),
            "processed_files": processed_files,
            "documents": final_doc_results,
            "total_fields": total_fields,
            "engine_used": engine_used
        })

    except Exception as e:
        traceback.print_exc()
        error_msg = str(e)
        return jsonify({
            "success": False,
            "error": f"Extraction failed: {error_msg}"
        }), 500

if __name__ == '__main__':
    print("=" * 60)
    print("  Document Scanner - Extract & Display")
    print("  Server running on http://127.0.0.1:5000")
    print("=" * 60)
    app.run(host='127.0.0.1', port=5000, debug=False)
