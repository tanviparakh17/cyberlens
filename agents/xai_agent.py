import numpy as np    # arrays handle karne ke liye
import shap            # SHAP library -- model ke decisions explain karne ke liye


def explain(prediction_result: dict, top_n: int = 5) -> dict:
    model = prediction_result["model"]           # url_agent se aaya trained model
    feature_row = prediction_result["feature_row"]  # is specific prediction ka feature row

    # TreeExplainer specifically tree-based models (jaise RandomForest) ke liye hai
    explainer = shap.TreeExplainer(model)

    # SHAP values calculate kiye -- har feature ka contribution number
    shap_values = explainer.shap_values(feature_row)

    # Different SHAP library versions alag shape mein data dete hain, isliye sab cases handle kar rahe hain:
    if isinstance(shap_values, list):
        # purana style: list hai [class0_values, class1_values]
        values = np.array(shap_values[1][0])   # class 1 (phishing) ke values liye
    else:
        arr = np.array(shap_values)
        if arr.ndim == 3:
            # shape: (samples, features, classes) -- sample 0, class 1 (phishing) liya
            values = arr[0, :, 1]
        elif arr.ndim == 2:
            # shape: (samples, features)
            values = arr[0]
        else:
            values = arr

    values = values.flatten()   # ensure kar rahe hain ki ye ek simple 1D list ban jaye
    feature_names = feature_row.columns.tolist()   # feature ke naam nikale

    # feature naam aur uska SHAP value ek saath jod diya
    contributions = list(zip(feature_names, values.tolist()))

    # sabse zyada impact wale (positive ya negative dono) upar la diye
    contributions.sort(key=lambda x: abs(x[1]), reverse=True)

    return {"top_features": contributions[:top_n]}   # sirf top N features return kiye