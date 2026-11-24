import os
import sys

# Configure environment for PaddleOCR before imports
CACHE_DIR = os.path.join(os.path.abspath(os.path.dirname(__file__)), ".cache")
os.makedirs(CACHE_DIR, exist_ok=True)
os.environ['USERPROFILE'] = CACHE_DIR
os.environ['HOME'] = CACHE_DIR
os.environ['PADDLE_HOME'] = os.path.join(CACHE_DIR, "paddle")
os.environ['PADDLEX_HOME'] = os.path.join(CACHE_DIR, "paddlex")
os.environ['PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK'] = 'True'
os.environ['FLAGS_use_onednn'] = '0'

# Patch paddlex runner to disable new_ir and mkldnn for smooth, error-free CPU inference
try:
    from paddlex.inference.models.runners.paddle_static import runner
    orig_create = runner.PaddleStaticRunner._create
    def patched_create(self):
        self._config['enable_new_ir'] = False
        self._config['run_mode'] = 'cpu'
        return orig_create(self)
    runner.PaddleStaticRunner._create = patched_create
except Exception:
    pass

import re
import cv2
import numpy as np
import fitz  # PyMuPDF
from PIL import Image, ImageOps

class OCRService:
    def __init__(self):
        self._ocr = None

    def _get_ocr(self):
        if self._ocr is None:
            from paddleocr import PaddleOCR
            # Optimize PaddleOCR: disable heavy unwarping (UVDoc) and orientation classify
            # Enable textline orientation for 90/180/270 degree text line recognition
            self._ocr = PaddleOCR(
                use_doc_unwarping=False,
                use_doc_orientation_classify=False,
                use_textline_orientation=True,
                lang='en'
            )
        return self._ocr

    def _parse_ocr_results(self, results):
        """
        Robustly parses PaddleOCR predict results across dictionary,
        OCRResult object, or list structures.
        """
        lines = []
        confidences = []

        if not results:
            return lines, confidences

        for res in results:
            rec_texts = []
            rec_scores = []

            if isinstance(res, dict):
                rec_texts = res.get('rec_texts', [])
                rec_scores = res.get('rec_scores', [])
            elif hasattr(res, 'rec_texts') and hasattr(res, 'rec_scores'):
                rec_texts = res.rec_texts
                rec_scores = res.rec_scores
            elif isinstance(res, list):
                # Standard PaddleOCR format: [[[box, (text, score)], ...]]
                for item in res:
                    if isinstance(item, (list, tuple)) and len(item) >= 2:
                        txt_score = item[1]
                        if isinstance(txt_score, (list, tuple)) and len(txt_score) >= 2:
                            rec_texts.append(str(txt_score[0]))
                            rec_scores.append(float(txt_score[1]))

            for t, s in zip(rec_texts, rec_scores):
                t_str = str(t).strip()
                if t_str:
                    lines.append(t_str)
                    try:
                        confidences.append(float(s))
                    except (ValueError, TypeError):
                        confidences.append(0.85)

        return lines, confidences

    def process_file(self, file_path):
        """
        Processes a PDF or image file.
        Returns:
            {
                "full_text": str,
                "pages": [
                    {
                        "page_number": int,
                        "text": str,
                        "confidence": float,
                        "method": "native_pdf" | "paddleocr"
                    }
                ],
                "page_count": int,
                "average_confidence": float
            }
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        ext = os.path.splitext(file_path)[1].lower()
        if ext == ".pdf":
            return self._process_pdf(file_path)
        elif ext in [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"]:
            return self._process_image(file_path)
        else:
            raise ValueError(f"Unsupported file format: {ext}")

    def _preprocess_image(self, image_path):
        """
        Opens the image with PIL, respects EXIF orientation transpose,
        converts to RGB, and resizes if extremely large for fast CPU OCR.
        """
        pil_img = Image.open(image_path)
        # Fix orientation from camera/phone EXIF metadata
        pil_img = ImageOps.exif_transpose(pil_img)
        # Ensure standard 3-channel RGB
        pil_img = pil_img.convert("RGB")

        # Proportional resize if image exceeds 2400px on any side
        max_dim = max(pil_img.size)
        if max_dim > 2400:
            scale = 2400.0 / max_dim
            new_size = (int(pil_img.size[0] * scale), int(pil_img.size[1] * scale))
            pil_img = pil_img.resize(new_size, Image.Resampling.LANCZOS)

        return np.array(pil_img)

    @staticmethod
    def _evaluate_text_quality(lines, scores):
        """
        Evaluates OCR quality for a given candidate rotation.
        Higher score = better, cleaner, more recognizable document text.
        Penalizes CJK artifacts (caused by slicing sideways across Latin letters).
        Rewards confidence, Latin alphanumeric density, and document keywords.
        """
        if not lines:
            return -100.0
        text = " ".join(lines)
        cjk_count = len(re.findall(r'[\u4e00-\u9fff]', text))
        avg_conf = sum(scores) / len(scores) if scores else 0.0

        words = re.findall(r'[a-zA-Z]{3,}', text)
        valid_word_chars = sum(len(w) for w in words)

        keywords = [
            'income', 'tax', 'department', 'govt', 'india', 'permanent', 'account', 'number',
            'father', 'name', 'date', 'birth', 'signature', 'male', 'female', 'dob', 'pan',
            'aadhaar', 'bank', 'branch', 'address', 'card', 'singh', 'republic', 'loan', 'salary'
        ]
        text_lower = text.lower()
        keyword_hits = sum(1 for kw in keywords if kw in text_lower)

        score = (avg_conf * 30.0) + (valid_word_chars * 0.5) + (keyword_hits * 15.0) - (cjk_count * 50.0)
        return score

    def _ocr_with_orientation(self, img_arr):
        """
        Performs OCR on an image array with automatic 4-way orientation detection.
        If the initial orientation is suspicious (CJK artifacts from vertical slicing,
        low word density, or portrait aspect ratio on an ID card), it tests 90/180/270 degree
        rotations and picks the one yielding the highest document quality score.
        """
        ocr = self._get_ocr()
        h, w = img_arr.shape[:2]

        # 1. Baseline OCR at 0 degrees
        results = ocr.predict(img_arr)
        best_lines, best_scores = self._parse_ocr_results(results)
        best_score = self._evaluate_text_quality(best_lines, best_scores)

        # Determine if orientation search is needed
        full_joined = " ".join(best_lines)
        has_cjk = bool(re.search(r'[\u4e00-\u9fff]', full_joined))
        needs_orientation_check = has_cjk or best_score < 140.0 or (h > w * 1.15 and best_score < 220.0)

        if needs_orientation_check:
            rotations = [
                (90, cv2.rotate(img_arr, cv2.ROTATE_90_CLOCKWISE)),
                (180, cv2.rotate(img_arr, cv2.ROTATE_180)),
                (270, cv2.rotate(img_arr, cv2.ROTATE_90_COUNTERCLOCKWISE)),
            ]
            for angle, rotated_img in rotations:
                r = ocr.predict(rotated_img)
                lines, scores = self._parse_ocr_results(r)
                q_score = self._evaluate_text_quality(lines, scores)
                if q_score > best_score:
                    best_score = q_score
                    best_lines = lines
                    best_scores = scores

        return best_lines, best_scores

    def _process_image(self, image_path):
        # Preprocess and obtain clean RGB numpy array
        img_arr = self._preprocess_image(image_path)

        page_text_lines, confidences = self._ocr_with_orientation(img_arr)

        full_text = "\n".join(page_text_lines)
        avg_conf = sum(confidences) / len(confidences) if confidences else 0.85

        return {
            "full_text": full_text,
            "pages": [
                {
                    "page_number": 1,
                    "text": full_text,
                    "confidence": round(avg_conf, 3),
                    "method": "paddleocr"
                }
            ],
            "page_count": 1,
            "average_confidence": round(avg_conf, 3)
        }

    def _process_pdf(self, pdf_path):
        doc = fitz.open(pdf_path)
        pages_info = []
        full_texts = []
        total_conf = 0.0

        for i, page in enumerate(doc):
            page_num = i + 1

            # 1. Try extracting native text first
            native_text = page.get_text("text").strip()

            # If native text is substantial and contains multiple words, it's digital
            if len(native_text) >= 40 and len(native_text.split()) >= 4:
                pages_info.append({
                    "page_number": page_num,
                    "text": native_text,
                    "confidence": 0.99,
                    "method": "native_pdf"
                })
                full_texts.append(f"--- PAGE {page_num} ---\n{native_text}")
                total_conf += 0.99
            else:
                # 2. Scanned PDF page: render in-memory pixmap at 150 DPI and run PaddleOCR with auto-orientation
                pix = page.get_pixmap(dpi=150)
                page_img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                page_arr = np.array(page_img)

                page_lines, page_scores = self._ocr_with_orientation(page_arr)

                p_text = "\n".join(page_lines)
                p_conf = sum(page_scores) / len(page_scores) if page_scores else 0.85

                pages_info.append({
                    "page_number": page_num,
                    "text": p_text,
                    "confidence": round(p_conf, 3),
                    "method": "paddleocr"
                })
                if p_text:
                    full_texts.append(f"--- PAGE {page_num} ---\n{p_text}")
                total_conf += p_conf

        doc.close()
        avg_conf = total_conf / len(pages_info) if pages_info else 0.0

        return {
            "full_text": "\n\n".join(full_texts),
            "pages": pages_info,
            "page_count": len(pages_info),
            "average_confidence": round(avg_conf, 3)
        }

if __name__ == "__main__":
    service = OCRService()
    test_file = os.path.join(os.path.dirname(__file__), "uploads", "1_74f34f95_test_pan.png")
    if os.path.exists(test_file):
        res = service.process_file(test_file)
        print("OCR Service test successful!")
        print("Extracted text:\n", res["full_text"])
        print("Confidence:", res["average_confidence"])
