import shap
import numpy as np

def get_shap_values(instance, model, feature_names):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(instance)
    return shap_values[1] if isinstance(shap_values, list) else shap_values

def generate_summary(instance, feature_names):
    summary = []

    for name, value in zip(feature_names, instance):
        # For Heart Disease features
        if name == "Cholesterol" and value > 250:
            summary.append("High cholesterol level may increase heart disease risk.")
        elif name == "RestingBP" and value > 140:
            summary.append("Elevated resting blood pressure is a known risk factor.")
        elif name == "Oldpeak" and value > 2:
            summary.append("Significant ST depression may indicate myocardial stress.")
        elif name == "MaxHR" and value < 100:
            summary.append("Low maximum heart rate could be a sign of poor cardiac function.")

        # For Diabetes features
        elif name == "Glucose" and value > 140:
            summary.append("High glucose level increases risk of diabetes.")
        elif name == "BMI" and value > 30:
            summary.append("High BMI is associated with type 2 diabetes.")
        elif name == "DiabetesPedigreeFunction" and value > 1:
            summary.append("Genetic history indicates higher risk of diabetes.")
        elif name == "Insulin" and value < 50:
            summary.append("Low insulin level may suggest insulin deficiency.")

    if not summary:
        summary.append("No major red flags in common indicators.")
    
    return summary
