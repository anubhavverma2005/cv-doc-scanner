import cv2
import numpy as np
import pytesseract
from flask import Flask, request, jsonify
from flask_cors import CORS
import base64
import os
import sys
import json

# --- CONFIGURATION ---
# Point to Tesseract executable (Windows Path)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Initialize Flask App
app = Flask(__name__)
CORS(app) # Allow React to communicate with this server

# --- HELPER FUNCTIONS ---

def detect_and_correct_skew(image):
    """
    Detect and correct document skew using PCA (Principal Component Analysis).
    Finds the orientation of text/edges and rotates image to straighten it.
    Only applies correction if tilt is detected (> 0.5 degrees).
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Step 1: Edge detection to find document boundaries
    edges = cv2.Canny(gray, 50, 150)
    
    # Step 2: Find contours/edges points
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    dilated = cv2.dilate(edges, kernel, iterations=2)
    
    # Step 3: Find all non-zero points (edge pixels)
    points = cv2.findNonZero(dilated)
    
    if points is None or len(points) < 100:
        print("Skew correction: insufficient edge points, skipping")
        return image
    
    # Step 4: Apply PCA to find principal axis (orientation)
    points = points.reshape(-1, 2).astype(np.float32)
    
    # Compute mean and covariance
    mean = np.mean(points, axis=0)
    centered_points = points - mean
    
    # Covariance matrix
    cov_matrix = np.cov(centered_points.T)
    
    # Get eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
    
    # Principal component (largest eigenvector)
    principal_axis = eigenvectors[:, np.argmax(eigenvalues)]
    
    # Step 5: Calculate rotation angle
    # Angle between principal axis and horizontal
    angle_rad = np.arctan2(principal_axis[1], principal_axis[0])
    angle_deg = np.degrees(angle_rad)
    
    # Normalize angle: we want horizontal document (angle close to 0 or 180)
    # If angle is in upper half (-180 to 0 or 0 to 180), adjust
    if angle_deg > 90:
        angle_deg -= 180
    elif angle_deg < -90:
        angle_deg += 180
    
    # Only correct if tilt is significant (> 0.5 degrees)
    if abs(angle_deg) < 0.5:
        print("Skew correction: image already straight, skipping")
        return image
    
    # Apply only 2 degree correction (very gentle)
    # Clamp the detected angle to ±2 degrees maximum
    angle_deg = np.clip(-angle_deg, -2, 2)
    
    print(f"Detected tilt: {angle_deg:.2f} degrees, applying correction")
    
    # Step 6: Rotate image to straighten
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    
    # Get rotation matrix
    M = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    
    # Rotate image
    rotated = cv2.warpAffine(
        image, M, (w, h),
        borderMode=cv2.BORDER_REFLECT_101,
        flags=cv2.INTER_CUBIC
    )
    
    print(f"Image rotated by {angle_deg:.2f} degrees")
    return rotated

def order_points(pts):
    """
    Orders coordinates in [Top-Left, Top-Right, Bottom-Right, Bottom-Left] order.
    Required for the perspective transform.
    Uses centroid-based polar sorting for robust corner ordering.
    """
    pts = np.array(pts, dtype="float32")
    
    # Compute the center of the contour
    center = pts.mean(axis=0)
    
    # Compute angles from center to each point
    angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
    
    # Sort points by angle (counter-clockwise from right)
    sorted_indices = np.argsort(angles)
    sorted_pts = pts[sorted_indices]
    
    # Reorder to start from top-left: find the point with smallest y, then smallest x
    # The sorted order is likely: right, bottom, left, top (or similar)
    # We want: top-left, top-right, bottom-right, bottom-left
    
    # Find the top-left (min y + min x among top candidates)
    min_y_idx = np.argmin(sorted_pts[:, 1])
    # Rotate so that top-left is first
    rect = np.roll(sorted_pts, -min_y_idx, axis=0)
    
    return rect

def four_point_transform(image, pts):
    """
    Warp the image based on the 4 detected points to get a flat 'scanned' look.
    """
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    # Compute width of new image (distance along top and bottom edges)
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    # Compute height of new image (distance along left and right edges)
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    # Ensure minimum dimensions to avoid empty images
    maxWidth = max(maxWidth, 50)
    maxHeight = max(maxHeight, 50)

    # Destination points in the output image
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    # Compute the perspective transform matrix and warp
    try:
        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
        print(f"Perspective warp successful: {image.shape} -> {warped.shape}")
        return warped
    except cv2.error as e:
        print(f"Perspective transform error: {e}")
        print(f"  rect shape: {rect.shape}, rect:\n{rect}")
        print(f"  dst shape: {dst.shape}, dst:\n{dst}")
        # Fallback: return original image if warp fails
        return image

def encode_image(image):
    """Convert OpenCV image to Base64 string for React."""
    _, buffer = cv2.imencode('.jpg', image)
    return base64.b64encode(buffer).decode('utf-8')

# --- PIPELINE STAGES ---

def stage1_preprocess(image):
    """
    Stage 1: Preprocessing
    Grayscale -> Bilateral Filter (smooth texture, keep edges) -> Canny Edge
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Bilateral filter removes noise but keeps edges sharp (essential for document detection)
    blur = cv2.bilateralFilter(gray, 9, 75, 75)
    # Canny Edge Detection with lower thresholds to catch more edges
    edges = cv2.Canny(blur, 50, 150)
    return gray, edges

def stage2_detection(edges, original_h, original_w):
    """
    Stage 2: Algorithmic Document Detection
    Finds the largest 4-sided contour. Includes fallbacks for poor detection.
    """
    # Morphological closing to bridge gaps in lines
    kernel = np.ones((5,5), np.uint8)
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    # Find contours
    cnts = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    
    # Sort by area (largest first)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    
    doc_cnt = None
    detected_area = 0

    # Loop over top 10 contours to find the document (not just top 5)
    for c in cnts[:10]:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)

        if len(approx) == 4:
            area = cv2.contourArea(c)
            # Prefer larger 4-sided contours; also check it's at least 10% of image
            min_area = (original_h * original_w) * 0.1
            if area > detected_area and area > min_area:
                doc_cnt = approx
                detected_area = area
    
    # If no 4-point contour found, use image boundaries as fallback
    if doc_cnt is None:
        print("No document contour found. Using full image boundaries.")
        doc_cnt = np.array([[[0, 0]], [[original_w, 0]], [[original_w, original_h]], [[0, original_h]]])
    else:
        print(f"Document detected with area: {detected_area} ({detected_area / (original_h * original_w) * 100:.1f}% of image)")
    
    return doc_cnt.reshape(4, 2)

def supports_dnn_cuda():
    """
    Quick runtime check whether OpenCV was built with DNN CUDA support and
    whether a CUDA device appears available. This is best-effort — the
    most reliable check is trying to use the DNN backend and catching errors.
    """
    try:
        bi = cv2.getBuildInformation().lower()
        # If OpenCV build doesn't mention cuda, it's unlikely DNN CUDA is available
        if 'cuda' not in bi:
            return False

        # Ensure the DNN constants exist
        if not (hasattr(cv2.dnn, 'DNN_BACKEND_CUDA') and hasattr(cv2.dnn, 'DNN_TARGET_CUDA')):
            return False

        # If cv2.cuda is present, check device count
        if hasattr(cv2, 'cuda'):
            try:
                cnt = cv2.cuda.getCudaEnabledDeviceCount()
            except Exception:
                cnt = 0
            if cnt <= 0:
                return False

        return True
    except Exception:
        return False


def stage4_enhancement(image):
    """
    Stage 4A: AI Super-Resolution (FSRCNN)
    Uses the RTX 4060 if CUDA is available in OpenCV build, otherwise CPU.
    """
    try:
        sr = cv2.dnn_superres.DnnSuperResImpl_create()
        
        # CHECK: Ensure you have 'FSRCNN_x4.pb' in the same folder as this script
        model_path = "FSRCNN_x4.pb"
        
        if os.path.exists(model_path):
            sr.readModel(model_path)
            sr.setModel("fsrcnn", 4) # 4x upscaling

            use_cuda = supports_dnn_cuda()
            if use_cuda:
                try:
                    sr.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
                    sr.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
                    print("Attempting to use CUDA for SuperRes...")
                except cv2.error as e:
                    # Setting backend/target may still raise if the OpenCV build
                    # doesn't support DNN CUDA even if CUDA exists — fall back.
                    print("Failed to enable DNN CUDA backend/target:", e)
                    use_cuda = False

            if not use_cuda:
                try:
                    sr.setPreferableBackend(cv2.dnn.DNN_BACKEND_DEFAULT)
                    sr.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
                    print("Using CPU for SuperRes.")
                except Exception:
                    # If setting fails, we'll still try upsample() which will use defaults
                    pass

            try:
                result = sr.upsample(image)
                return result
            except cv2.error as e:
                print("SuperRes upsample failed with OpenCV error:", e)
                # Try CPU fallback if CUDA path failed
                try:
                    sr.setPreferableBackend(cv2.dnn.DNN_BACKEND_DEFAULT)
                    sr.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
                    result = sr.upsample(image)
                    return result
                except Exception as e2:
                    print("CPU SuperRes also failed. Falling back to bicubic. Error:", e2)
                    h, w = image.shape[:2]
                    return cv2.resize(image, (w*4, h*4), interpolation=cv2.INTER_CUBIC)
        else:
            print("FSRCNN model not found. Falling back to Bicubic.")
            # Fallback to standard Bicubic if model missing
            h, w = image.shape[:2]
            return cv2.resize(image, (w*4, h*4), interpolation=cv2.INTER_CUBIC)
    except Exception as e:
        print(f"SuperRes Error: {e}")
        return image

def stage4_binarization(image):
    """
    Stage 4B: Binarization + OCR preprocessing
    Simplified, balanced approach for best OCR results.
    Focus: preserve text detail while improving contrast.
    """
    # Ensure grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    print(f"Initial: brightness={np.mean(gray):.1f}, contrast={np.std(gray)/(np.mean(gray)+1e-6):.2f}")
    
    # Step 1: Bilateral filter to smooth while preserving edges (no aggressive denoising)
    bilateral = cv2.bilateralFilter(gray, 9, 75, 75)
    
    # Step 2: Apply moderate CLAHE for contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(10, 10))
    enhanced = clahe.apply(bilateral)
    
    print(f"After CLAHE: brightness={np.mean(enhanced):.1f}, contrast={np.std(enhanced)/(np.mean(enhanced)+1e-6):.2f}")
    
    # Step 3: Gamma correction only if truly overexposed
    mean_val = np.mean(enhanced)
    if mean_val > 200:
        gamma = 1.8
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        enhanced = cv2.LUT(enhanced, table)
        print(f"Gamma correction applied: brightness now {np.mean(enhanced):.1f}")
    
    # Step 4: Moderate unsharp mask for text sharpness (not too aggressive)
    gaussian = cv2.GaussianBlur(enhanced, (3, 3), 1.0)
    sharpened = cv2.addWeighted(enhanced, 1.3, gaussian, -0.3, 0)
    
    # Step 5: Adaptive thresholding with moderate parameters
    binary = cv2.adaptiveThreshold(
        sharpened, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 13, 4
    )
    
    # Step 6: Light morphological cleanup (minimal)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
    
    # Step 7: Auto-invert if needed
    if np.mean(binary) < 128:
        binary = cv2.bitwise_not(binary)
        print("Image inverted")
    
    print(f"Final binary: white_ratio={np.sum(binary == 255) / binary.size * 100:.1f}%")
    return binary

def calculate_image_quality_metrics(gray_image):
    """
    Calculate quality metrics for preprocessing evaluation.
    Returns a dict with contrast, sharpness, and noise estimates.
    """
    metrics = {}
    
    # Normalize to 0-1 range for better interpretation
    gray_norm = gray_image.astype(np.float32) / 255.0
    
    # Contrast (standard deviation normalized by mean)
    mean_val = np.mean(gray_norm)
    std_val = np.std(gray_norm)
    contrast = std_val / (mean_val + 1e-6)
    metrics["contrast"] = float(contrast)
    
    # Sharpness (Laplacian variance)
    laplacian = cv2.Laplacian(gray_image, cv2.CV_64F)
    sharpness = float(np.var(laplacian))
    metrics["sharpness"] = sharpness
    
    # Brightness (mean pixel value 0-255)
    brightness = float(np.mean(gray_image))
    metrics["brightness"] = brightness
    
    # Additional metrics for better evaluation
    # Entropy (measure of information content)
    hist = cv2.calcHist([gray_image], [0], None, [256], [0, 256])
    hist = hist.flatten() / hist.sum()
    entropy = -np.sum(hist * np.log2(hist + 1e-10))
    metrics["entropy"] = float(entropy)
    
    # Brightness quality score (ideal range 80-150)
    if 80 <= brightness <= 150:
        brightness_score = 100.0
    elif brightness < 80:
        brightness_score = max(0, 100 - (80 - brightness) / 2)
    else:
        brightness_score = max(0, 100 - (brightness - 150) / 2)
    metrics["brightness_quality_score"] = float(brightness_score)
    
    return metrics

def extract_ocr_metrics(binary_image):
    """
    Extract OCR confidence and quality metrics from Tesseract.
    Returns a dict with character confidence, word count, and confidence scores.
    """
    try:
        # Get detailed OCR data with per-character confidence
        data = pytesseract.image_to_data(binary_image, config='--psm 6', output_type=pytesseract.Output.DICT)
        
        # Extract confidence scores (skip invalid entries with -1)
        confidences = [int(conf) for conf in data['conf'] if int(conf) > 0]
        
        metrics = {}
        if confidences:
            metrics["mean_confidence"] = float(np.mean(confidences))
            metrics["min_confidence"] = float(np.min(confidences))
            metrics["max_confidence"] = float(np.max(confidences))
            metrics["confidence_std"] = float(np.std(confidences))
            metrics["characters_detected"] = len(confidences)
        else:
            metrics["mean_confidence"] = 0.0
            metrics["min_confidence"] = 0.0
            metrics["max_confidence"] = 0.0
            metrics["confidence_std"] = 0.0
            metrics["characters_detected"] = 0
        
        # Count words (non-empty text entries)
        words = [w for w in data['text'] if w.strip()]
        metrics["words_detected"] = len(words)
        
        return metrics
    except Exception as e:
        print(f"Error extracting OCR metrics: {e}")
        return {
            "mean_confidence": 0.0,
            "min_confidence": 0.0,
            "max_confidence": 0.0,
            "confidence_std": 0.0,
            "characters_detected": 0,
            "words_detected": 0,
            "error": str(e)
        }

# --- DIAGNOSTICS ENDPOINT ---

@app.route('/diagnostics', methods=['GET'])
def diagnostics():
    """
    Returns OpenCV build info and GPU/CUDA availability for debugging.
    """
    info = {
        "cuda_supported": supports_dnn_cuda(),
        "dnn_cuda_constants": {
            "DNN_BACKEND_CUDA": hasattr(cv2.dnn, 'DNN_BACKEND_CUDA'),
            "DNN_TARGET_CUDA": hasattr(cv2.dnn, 'DNN_TARGET_CUDA')
        },
        "cv2_cuda_module": hasattr(cv2, 'cuda')
    }
    
    if hasattr(cv2, 'cuda'):
        try:
            info["cuda_device_count"] = cv2.cuda.getCudaEnabledDeviceCount()
        except Exception as e:
            info["cuda_device_count_error"] = str(e)
    
    # Build info snippet
    try:
        build_info = cv2.getBuildInformation()
        # Extract CUDA line if present
        for line in build_info.split('\n'):
            if 'cuda' in line.lower():
                info["build_cuda_line"] = line.strip()
                break
    except Exception as e:
        info["build_info_error"] = str(e)
    
    return jsonify(info)

# --- MAIN API ROUTE ---

@app.route('/process-document', methods=['POST'])
def process_document():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    
    # 1. Read Image
    file_bytes = np.frombuffer(file.read(), np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    h, w = image.shape[:2]

    try:
        # STAGE 0: Skew Correction (NEW)
        # Detect and straighten tilted documents using PCA
        corrected = detect_and_correct_skew(image)
        
        # STAGE 1: Preprocessing
        gray, edges = stage1_preprocess(corrected)

        # STAGE 2: Detection
        doc_cnt = stage2_detection(edges, h, w)

        # STAGE 3: Perspective Transform (The "Scan")
        warped = four_point_transform(corrected, doc_cnt)

        # STAGE 4A: Enhancement (Super Res)
        # We resize the warped image slightly down before super-res to save time if it's huge,
        # or pass raw if you have a powerful GPU. Let's pass raw.
        enhanced = stage4_enhancement(warped)

        # STAGE 4B: Binarization
        binary = stage4_binarization(enhanced)

        # STAGE 5: OCR
        # Use PSM 6 (uniform text block) - best for documents
        text_data = pytesseract.image_to_string(binary, config='--psm 6')
        ocr_metrics = extract_ocr_metrics(binary)
        
        current_confidence = ocr_metrics.get('mean_confidence', 0)
        text_length = len(text_data.strip())
        
        print(f"OCR: {text_length} chars, confidence {current_confidence:.1f}")
        if text_length == 0:
            print("WARNING: OCR returned empty text.")

        # Calculate quality metrics on enhanced image
        preprocessing_metrics = calculate_image_quality_metrics(enhanced)
        
        print(f"Preprocessing: {preprocessing_metrics}")
        print(f"OCR: {ocr_metrics}")

        # Prepare Response
        response_data = {
            "original": encode_image(image),
            "corrected": encode_image(corrected),
            "edges": encode_image(edges),
            "scanned": encode_image(warped),
            "enhanced": encode_image(enhanced),
            "binary": encode_image(binary),
            "text": text_data,
            "metrics": {
                "preprocessing": preprocessing_metrics,
                "ocr": ocr_metrics
            }
        }

        return jsonify(response_data)

    except Exception as e:
        print(e)
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
