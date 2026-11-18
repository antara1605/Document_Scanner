# Document Scanner & AI Information Extractor (`dscan`)

An intelligent, local-first web application for automated document OCR and structured information extraction. Upload any document—identity proofs, driving licenses, bank cheques, survey reports, deeds, discharge vouchers, invoices, and receipts—and extract clean, structured data and summaries in seconds using **Groq AI** and high-accuracy local OCR engines.

---

## 1. Key Features

- **Universal Multi-Document Upload:**
  - Drag-and-drop batch upload for **PDF** (single and multi-page), **JPG**, **JPEG**, **PNG**, **WEBP**, **BMP**, and **TIFF**.
  - Processes multiple documents in a single unified workflow.

- **High-Accuracy Local OCR:**
  - Multipage PDF rendering and native digital text extraction via **PyMuPDF (`fitz`)**.
  - Scanned document text extraction powered by **PaddleOCR** and **EasyOCR** running locally on CPU.
  - Image preprocessing and enhancement via **OpenCV (`cv2`)** (contrast normalization, grayscale conversion, and auto-deskew).

- **Intelligent Structured AI Extraction:**
  - Powered by **Groq Cloud API** with rapid inference (supports models like `llama-3.3-70b-versatile`, `qwen/qwen3.8-27b`, etc.).
  - Extracts clean, human-readable key-value fields and document categories:
    - **Identity & KYC:** Full Name, Father/Spouse Name, DOB, Gender, PAN, Aadhaar, Voter ID, Passport.
    - **Driving License:** License Number, Validity, Issue Date, Issuing Authority, Vehicle Classes (LMV, MCWG).
    - **Banking & Finance:** Bank Name, Branch, Account Number, IFSC Code, Cheque Number, Salary, Deductions.
    - **Motor Survey & Insurance:** Policy Number, Claim Number, Vehicle Registration No, Chassis/Engine No, Loss Date, Surveyor Name.
    - **Property & Legal:** Deed Number, Plot/Dag No, Khatian No, Mouza, Area, Sub-Registry Office.
  - Generates clear document summaries and filters out empty/null placeholders.

- **Built-In Offline Fallback Engine:**
  - If no Groq API key is configured or when offline, the system automatically falls back to an internal **Rule & Heuristic Engine** using regex patterns and document classifiers.

- **Modern Interactive Web UI:**
  - Single-page responsive interface (Bootstrap 5 & FontAwesome).
  - Live progress stepper during upload, OCR, and AI extraction.
  - Interactive document cards with document-type badges and concise summaries.
  - Key-value field tables with copy-to-clipboard buttons.
  - Full raw OCR text inspection modal for verification.

---

## 2. Project Structure

```
dscan/
├── app.py              # Main Flask web application and API endpoints
├── ai_extractor.py     # Groq AI extraction client and offline rule-based fallback
├── ocr_service.py      # PyMuPDF and local PaddleOCR/EasyOCR document engine
├── start.bat           # One-click Windows startup script
├── templates/
│   └── index.html      # Interactive frontend web dashboard
├── uploads/            # Temporary disk storage for documents during OCR
├── .cache/             # Cached model weights and temporary files
├── .env                # Environment configuration (API keys, models)
└── README.md           # Documentation
```

---

## 3. Configuration (`.env`)

Create or update `.env` in the root project folder:

```ini
# Groq AI Configuration
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=llama-3.3-70b-versatile

# Application Settings
SECRET_KEY=dscan-extract-2026
UPLOAD_DIR=d:/dscan/uploads
CACHE_DIR=d:/dscan/.cache

# PaddleOCR Settings
PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True
```

> **Note:** Even without a valid Groq API key, the system automatically runs the offline rule-based extractor.

---

## 4. How to Run

### Method 1: Using the Batch File (Windows)
Double-click `start.bat` in the project root. This script will:
1. Verify Python availability.
2. Free port `5000` if previously in use.
3. Launch the Flask web server.
4. Automatically open `http://127.0.0.1:5000` in your default browser.

### Method 2: Manual Terminal Startup
Open PowerShell or Command Prompt in the project directory:

```powershell
# Activate your virtual environment if applicable, then run:
python app.py
```

Then navigate to:
```
http://127.0.0.1:5000
```

---

## 5. API Reference

### `GET /`
Renders and serves the interactive single-page web dashboard.

---

### `POST /upload-and-process`
Uploads and processes one or more document files.

- **Request:** `multipart/form-data`
  - Form field: `documents` (one or multiple files: PDF, JPG, PNG, etc.)
- **Response Example (`application/json`):**
```json
{
  "success": true,
  "processed_files_count": 1,
  "processed_files": ["DL_STATUS.pdf"],
  "total_fields": 8,
  "engine_used": "Groq AI (llama-3.3-70b-versatile)",
  "documents": [
    {
      "document_name": "DL_STATUS.pdf",
      "document_type": "Driving License",
      "summary": "Driving License of SUBRATA MUKHERJEE valid up to 2035.",
      "page_count": 1,
      "confidence": 0.92,
      "fields": {
        "Full Name": "SUBRATA MUKHERJEE",
        "License Number": "WB58 20150001234",
        "Date of Birth": "15/08/1985",
        "Validity": "14/08/2035",
        "Vehicle Class": "LMV, MCWG"
      },
      "ocr_text": "Extracted OCR text content..."
    }
  ]
}
```

---

## 6. Privacy & Local Storage

- **Local File Processing:** Uploaded files are temporarily stored inside the local `uploads/` folder so the OCR engine can read them from disk.
- **No Cloud Document Storage:** Original document files remain local to your machine. Only extracted text is sent to the Groq API endpoint for field parsing (or processed locally when offline).
- **Disk Cleanup:** You can safely delete old files inside `uploads/` at any time to free up disk space.
