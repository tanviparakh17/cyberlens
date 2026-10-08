import os
import cv2
import pandas as pd
from agents.url_agent import predict as predict_url


def decode_qr_code(image_path: str) -> str:
    """
    Decodes a QR code image file into a URL string using OpenCV.
    Returns None if no valid QR code is found.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found at: {image_path}")

    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Failed to read image at: {image_path}")

    detector = cv2.QRCodeDetector()
    decoded_text, bbox, _ = detector.detectAndDecode(img)

    if not decoded_text:
        return None

    return decoded_text.strip()


def predict(qr_image_path: str) -> dict:
    """
    Decodes a QR code image and forwards the URL to url_agent.predict().
    Returns standard CyberLens interface dict compatible with XAI and Report agents.
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

    # Delegate URL analysis to existing url_agent
    url_result = predict_url(decoded_url)

    # Attach QR-specific fields to output
    url_result["qr_image_path"] = qr_image_path
    url_result["decoded_url"] = decoded_url

    return url_result


if __name__ == "__main__":
    import qrcode

    os.makedirs("data", exist_ok=True)
    test_qr_path = "data/test_phishing_qr.png"
    phishing_url = "http://paypal-account-verify.suspend-secure-login.xyz"

    # Generate a sample phishing QR image for instant testing
    qr_img = qrcode.make(phishing_url)
    qr_img.save(test_qr_path)
    print(f"✅ Generated sample QR code image at {test_qr_path}")

    # Run QR Agent prediction
    result = predict(test_qr_path)
    print("\n--- QR AGENT TEST OUTPUT ---")
    print(f"Image Path:  {result['qr_image_path']}")
    print(f"Decoded URL: {result['decoded_url']}")
    print(f"Verdict:     {result['verdict']}")
    print(f"Confidence:  {result['confidence']}")
