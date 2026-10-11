import os
import time

import cv2
import pandas as pd

from agents.url_agent import predict as predict_url

# Longest image side (pixels) to try when shrinking. Screen photos often have
# moire patterns that disappear once the image is made smaller.
DOWNSCALE_TARGETS = (1600, 1200, 900, 700, 500)

# Images with a longer side than this are tried in shrunk form first (full-size
# detection on a 12 MP photo is slow), and at full size only as a last resort.
LARGE_IMAGE_SIDE = 1600

# Give up after this many seconds so one hard image cannot hold a server worker.
MAX_SECONDS = 8.0


def _gray(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img


def _variants(img):
    """
    Yields (name, image) versions of the picture, cheapest first. The caller stops at
    the first version that decodes. Large photos are tried shrunk first and at full
    size only at the very end.
    """
    gray = _gray(img)
    h, w = gray.shape[:2]
    longest = max(h, w)
    is_large = longest > LARGE_IMAGE_SIDE

    if not is_large:
        yield "original", img
        yield "gray", gray

    # Shrunk copies (INTER_AREA averages pixels, which removes screen moire)
    for target in DOWNSCALE_TARGETS:
        if longest > target:
            scale = target / longest
            size = (max(1, int(w * scale)), max(1, int(h * scale)))
            yield f"downscale_{target}", cv2.resize(gray, size, interpolation=cv2.INTER_AREA)

    # Enlarged copies for very small images
    if longest < 500:
        for factor in (2, 3):
            yield f"upscale_x{factor}", cv2.resize(
                gray, (w * factor, h * factor), interpolation=cv2.INTER_CUBIC
            )

    # Light blur and a hard black/white version, both on a mid-size copy
    mid = gray
    if longest > 1000:
        scale = 1000 / longest
        mid = cv2.resize(gray, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    blurred = cv2.GaussianBlur(mid, (3, 3), 0)
    yield "blur", blurred
    _, otsu = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    yield "otsu", otsu

    if is_large:
        yield "original", img
        yield "gray", gray


def _detectors():
    """The standard OpenCV detector, plus the Aruco-based one when this OpenCV has it."""
    detectors = [("standard", cv2.QRCodeDetector())]
    if hasattr(cv2, "QRCodeDetectorAruco"):
        detectors.append(("aruco", cv2.QRCodeDetectorAruco()))
    return detectors


def _with_border(img):
    """White margin around the picture (a QR code needs a blank 'quiet zone')."""
    pad = max(20, int(max(img.shape[:2]) * 0.08))
    value = 255 if img.ndim == 2 else (255, 255, 255)
    return cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=value)


def decode_qr_code(image_path: str) -> str:
    """
    Decodes a QR code image file into a string using OpenCV.
    Tries several image versions (shrunk, enlarged, with a white border, thresholded)
    and two detectors, and returns the first successful result.
    Returns None if no valid QR code is found.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found at: {image_path}")

    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Failed to read image at: {image_path}")

    detectors = _detectors()
    deadline = time.monotonic() + MAX_SECONDS

    for _variant_name, variant in _variants(img):
        if time.monotonic() > deadline:
            break
        for candidate in (variant, _with_border(variant)):
            for _detector_name, detector in detectors:
                try:
                    text, _points, _straight = detector.detectAndDecode(candidate)
                except cv2.error:
                    continue
                if text:
                    return text.strip()

    return None


def predict(qr_image_path: str) -> dict:
    """
    Decodes a QR code image and forwards the URL to url_agent.predict().
    Returns the standard CyberLens interface dict compatible with the XAI and report agents.
    """
    decoded_url = decode_qr_code(qr_image_path)

    if not decoded_url:
        return {
            "qr_image_path": qr_image_path,
            "decoded_url": None,
            "verdict": "unknown",
            "confidence": 0.0,
            "features": {"qr_decoded": False},
            "model": None,
            "feature_row": pd.DataFrame(),
            "error": "No valid QR code detected in image",
        }

    # Delegate URL analysis to the existing url_agent
    url_result = predict_url(decoded_url)

    # Attach QR-specific fields to the output
    url_result["qr_image_path"] = qr_image_path
    url_result["decoded_url"] = decoded_url

    return url_result


if __name__ == "__main__":
    # Usage: python -m agents.qr_agent path\to\image.png
    # Without an argument, a sample phishing QR is generated and tested.
    import sys

    if len(sys.argv) > 1:
        test_qr_path = sys.argv[1]
    else:
        import qrcode

        os.makedirs("data", exist_ok=True)
        test_qr_path = "data/test_phishing_qr.png"
        qrcode.make("http://paypal-account-verify.suspend-secure-login.xyz").save(test_qr_path)
        print(f"Generated sample QR code image at {test_qr_path}")

    result = predict(test_qr_path)
    print("\n--- QR AGENT TEST OUTPUT ---")
    print(f"Image Path:  {result['qr_image_path']}")
    print(f"Decoded URL: {result['decoded_url']}")
    print(f"Verdict:     {result['verdict']}")
    print(f"Confidence:  {result['confidence']}")