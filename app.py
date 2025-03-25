import os
from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load trained models
models = {
    "Naive Bayes": joblib.load("naive_bayes_model.pkl"),
    "KNN": joblib.load("knn_model.pkl"),
    "SVM": joblib.load("svm_model.pkl"),
    "Random Forest": joblib.load("random_forest_model.pkl")
}

# Load preprocessing tools
scaler = joblib.load("scaler.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# Load model performance metrics from CSV
metrics_df = pd.read_csv("model_performance_metrics.csv", index_col=0)
metrics_dict = metrics_df.to_dict(orient="index")

# Define feature names
FEATURE_NAMES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]

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
        X_scaled = scaler.transform(df_input)

        # Get predictions
        predictions = {}
        for model_name, model in models.items():
            y_pred = model.predict(X_scaled)
            y_pred_label = label_encoder.inverse_transform(y_pred)[0]
            predictions[model_name] = y_pred_label

        # Get the best-performing model based on Accuracy
        best_model = metrics_df["Accuracy"].idxmax()
        best_model_accuracy = metrics_df.loc[best_model, "Accuracy"]

        # Include all additional performance metrics
        best_model_metrics = metrics_dict[best_model]
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
            "final_recommendation": f"The best performing model is '{best_model}' with an accuracy of {best_model_accuracy:.2%}.",
            "additional_metrics": additional_metrics
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8081))  # Use 8081 instead of 5000
    app.run(host='0.0.0.0', port=port)
