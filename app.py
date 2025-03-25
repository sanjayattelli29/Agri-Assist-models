import os
from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

app = Flask(__name__)

# Function to safely load model files
def load_file(file_name):
    if os.path.exists(file_name):
        return joblib.load(file_name)
    else:
        print(f"⚠️ Warning: {file_name} not found!")
        return None  # Avoids crashing

# Load trained models
models = {
    "Naive Bayes": load_file("naive_bayes_model.pkl"),
    "KNN": load_file("knn_model.pkl"),
    "SVM": load_file("svm_model.pkl"),
    "Random Forest": load_file("random_forest_model.pkl")
}

# Load preprocessing tools
scaler = load_file("scaler.pkl")
label_encoder = load_file("label_encoder.pkl")

# Load model performance metrics
if os.path.exists("model_performance_metrics.csv"):
    metrics_df = pd.read_csv("model_performance_metrics.csv", index_col=0)
    metrics_dict = metrics_df.to_dict(orient="index")
else:
    print("⚠️ Warning: model_performance_metrics.csv not found!")
    metrics_dict = {}

# Define feature names
FEATURE_NAMES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]

# Home route to prevent 404 on Render
@app.route('/')
def home():
    return jsonify({"message": "Air Quality Prediction API is running!"})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json

        # Validate input
        if "features" not in data or not isinstance(data["features"], list) or len(data["features"]) != 7:
            return jsonify({"error": "Invalid input. Expected 7 feature values in a list."}), 400

        # Convert input to DataFrame
        df_input = pd.DataFrame([data["features"]], columns=FEATURE_NAMES)

        # Scale input
        if scaler:
            X_scaled = scaler.transform(df_input)
        else:
            return jsonify({"error": "Scaler is missing!"}), 500

        # Get predictions & probability scores for ROC Curve
        predictions = {}
        roc_data = {}
        y_true = data.get("y_true", [])  # Ensure API receives actual labels for ROC

        for model_name, model in models.items():
            if model:
                y_pred = model.predict(X_scaled)
                y_pred_label = label_encoder.inverse_transform(y_pred)[0] if label_encoder else "Unknown"
                predictions[model_name] = y_pred_label

                # Get probability scores if available (for ROC)
                if hasattr(model, "predict_proba"):
                    y_scores = model.predict_proba(X_scaled)[:, 1]  # Use positive class scores
                    roc_data[model_name] = {"y_scores": y_scores.tolist()}
            else:
                predictions[model_name] = "Model not available"

        # Compute ROC Curve if actual labels (`y_true`) are provided
        if y_true:
            for model_name, roc_info in roc_data.items():
                y_scores = roc_info["y_scores"]
                fpr, tpr, _ = roc_curve(y_true, y_scores)
                roc_data[model_name]["FPR"] = fpr.tolist()
                roc_data[model_name]["TPR"] = tpr.tolist()

        # Get the best-performing model based on Accuracy
        best_model = metrics_df["Accuracy"].idxmax() if not metrics_df.empty else "Unknown"
        best_model_accuracy = metrics_df.loc[best_model, "Accuracy"] if best_model in metrics_df.index else "N/A"

        # Include all additional performance metrics
        best_model_metrics = metrics_dict.get(best_model, {})
        additional_metrics = {
            "Accuracy": best_model_metrics.get("Accuracy", "N/A"),
            "Precision": best_model_metrics.get("Precision", "N/A"),
            "Recall": best_model_metrics.get("Recall", "N/A"),
            "F1-Score": best_model_metrics.get("F1-Score", "N/A"),
            "ROC-AUC": best_model_metrics.get("ROC-AUC", "N/A"),
            "Log Loss": best_model_metrics.get("Log Loss", "N/A"),
            "MAE": best_model_metrics.get("MAE", "N/A"),
            "MSE": best_model_metrics.get("MSE", "N/A"),
            "RMSE": best_model_metrics.get("RMSE", "N/A"),
            "R² Score": best_model_metrics.get("R² Score", "N/A"),
            "Cohen’s Kappa": best_model_metrics.get("Cohen’s Kappa", "N/A"),
            "Balanced Accuracy": best_model_metrics.get("Balanced Accuracy", "N/A"),
            "Hinge Loss": best_model_metrics.get("Hinge Loss", "N/A"),
            "Gini Coefficient": best_model_metrics.get("Gini Coefficient", "N/A"),
            "Chi-Square (Z²)": best_model_metrics.get("Chi-Square (Z²)", "N/A")
        }

        return jsonify({
            "predictions": predictions,
            "metrics": metrics_dict,
            "roc_curve": roc_data,  # ✅ Added ROC curve data
            "final_recommendation": f"The best performing model is '{best_model}' with an accuracy of {best_model_accuracy}.",
            "additional_metrics": additional_metrics
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))  # Use PORT from Render
    app.run(host='0.0.0.0', port=port)
