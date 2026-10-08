"""
CyberLens APK Malware Detection Agent

Analyzes Android APK files using Androguard and extracts:
- Package name
- Application name
- Permissions
- Suspicious permissions
- DEX method/class information
- APK SHA-256 hash
- Basic risk score
- Evidence

Usage:
    python -m agents.apk_agent "path\\to\\file.apk"
"""

import hashlib
import os
import sys

from androguard.misc import AnalyzeAPK


# Permissions that can be useful as indicators during APK analysis.
# These alone do NOT prove that an APK is malicious.
RISKY_PERMISSIONS = {
    "android.permission.READ_SMS",
    "android.permission.RECEIVE_SMS",
    "android.permission.SEND_SMS",
    "android.permission.READ_CONTACTS",
    "android.permission.WRITE_CONTACTS",
    "android.permission.READ_CALL_LOG",
    "android.permission.WRITE_CALL_LOG",
    "android.permission.RECORD_AUDIO",
    "android.permission.CAMERA",
    "android.permission.ACCESS_FINE_LOCATION",
    "android.permission.ACCESS_COARSE_LOCATION",
    "android.permission.READ_PHONE_STATE",
    "android.permission.CALL_PHONE",
    "android.permission.READ_EXTERNAL_STORAGE",
    "android.permission.WRITE_EXTERNAL_STORAGE",
    "android.permission.REQUEST_INSTALL_PACKAGES",
    "android.permission.SYSTEM_ALERT_WINDOW",
    "android.permission.RECEIVE_BOOT_COMPLETED",
}


def calculate_sha256(file_path):
    """Calculate SHA-256 hash of the APK."""
    sha256 = hashlib.sha256()

    with open(file_path, "rb") as file:
        while True:
            chunk = file.read(1024 * 1024)

            if not chunk:
                break

            sha256.update(chunk)

    return sha256.hexdigest()


def analyze_apk(apk_path):
    """
    Analyze an Android APK and return a structured CyberLens result.
    """

    if not os.path.isfile(apk_path):
        raise FileNotFoundError(f"APK file not found: {apk_path}")

    if not apk_path.lower().endswith(".apk"):
        raise ValueError("The supplied file is not an APK.")

    print(f"[APK Agent] Analyzing: {apk_path}")

    # Androguard analysis
    apk, dex_objects, analysis = AnalyzeAPK(apk_path)

    # ---------------------------------------------------------
    # Basic APK information
    # ---------------------------------------------------------

    package_name = apk.get_package()

    try:
        app_name = apk.get_app_name()
    except Exception:
        app_name = None

    # ---------------------------------------------------------
    # Permissions
    # ---------------------------------------------------------

    permissions = []

    try:
        permissions = apk.get_permissions() or []
    except Exception:
        permissions = []

    permissions = sorted(set(permissions))

    risky_permissions = [
        permission
        for permission in permissions
        if permission in RISKY_PERMISSIONS
    ]

    # ---------------------------------------------------------
    # DEX / code information
    # ---------------------------------------------------------

    class_names = []
    method_names = []

    for dex in dex_objects:
        try:
            classes = dex.get_classes()

            for clazz in classes:
                class_name = clazz.get_name()

                if class_name:
                    class_names.append(class_name)

                try:
                    for method in clazz.get_methods():
                        method_name = method.get_name()

                        if method_name:
                            method_names.append(method_name)

                except Exception:
                    pass

        except Exception:
            pass

    class_names = sorted(set(class_names))
    method_names = sorted(set(method_names))

    # ---------------------------------------------------------
    # APK hash
    # ---------------------------------------------------------

    sha256 = calculate_sha256(apk_path)

    # ---------------------------------------------------------
    # Basic heuristic risk scoring
    # ---------------------------------------------------------

    risk_score = 0.0
    evidence = []

    # Risk from suspicious permissions
    permission_score = min(len(risky_permissions) * 0.08, 0.40)

    if risky_permissions:
        risk_score += permission_score

        evidence.append(
            f"{len(risky_permissions)} potentially sensitive permission(s) detected."
        )

    # Risk from very large number of permissions
    if len(permissions) >= 20:
        risk_score += 0.15

        evidence.append(
            f"APK requests a relatively large number of permissions ({len(permissions)})."
        )

    # Risk from large method count
    if len(method_names) >= 5000:
        risk_score += 0.15

        evidence.append(
            f"APK contains a large number of methods ({len(method_names)})."
        )

    # Cap score
    risk_score = min(risk_score, 1.0)

    # ---------------------------------------------------------
    # Verdict
    # ---------------------------------------------------------

    if risk_score >= 0.70:
        verdict = "malicious"
    elif risk_score >= 0.35:
        verdict = "suspicious"
    else:
        verdict = "safe"

    if not evidence:
        evidence.append(
            "No strong indicators were detected by the current heuristic checks."
        )

    # ---------------------------------------------------------
    # Standardized CyberLens result
    # ---------------------------------------------------------

    result = {
        "agent": "apk",
        "input_type": "android_apk",
        "apk_path": os.path.abspath(apk_path),

        "package_name": package_name,
        "app_name": app_name,

        "verdict": verdict,
        "risk_score": round(risk_score, 3),

        "permissions": permissions,
        "risky_permissions": risky_permissions,

        "class_count": len(class_names),
        "method_count": len(method_names),

        # Keep the result manageable.
        "classes": class_names[:100],
        "methods": method_names[:100],

        "sha256": sha256,

        "evidence": evidence,
    }

    return result


def print_result(result):
    """Print the APK analysis result in a readable format."""

    print("\n" + "=" * 60)
    print("CYBERLENS APK ANALYSIS RESULT")
    print("=" * 60)

    print(f"APK           : {result['apk_path']}")
    print(f"App Name      : {result['app_name']}")
    print(f"Package       : {result['package_name']}")

    print(f"\nVerdict       : {result['verdict'].upper()}")
    print(f"Risk Score    : {result['risk_score']}")

    print(f"\nPermissions   : {len(result['permissions'])}")
    print(f"Risky Perms   : {len(result['risky_permissions'])}")

    print(f"Classes       : {result['class_count']}")
    print(f"Methods       : {result['method_count']}")

    print(f"\nSHA-256:")
    print(result["sha256"])

    print("\nRisky Permissions:")

    if result["risky_permissions"]:
        for permission in result["risky_permissions"]:
            print(f"  - {permission}")
    else:
        print("  None detected")

    print("\nEvidence:")

    for item in result["evidence"]:
        print(f"  - {item}")

    print("=" * 60)


def main():
    """Command-line entry point."""

    if len(sys.argv) != 2:
        print(
            "Usage:\n"
            'python -m agents.apk_agent "path\\to\\file.apk"'
        )
        return

    apk_path = sys.argv[1]

    try:
        result = analyze_apk(apk_path)
        print_result(result)

    except Exception as error:
        print("\n[APK Agent ERROR]")
        print(str(error))


if __name__ == "__main__":
    main()