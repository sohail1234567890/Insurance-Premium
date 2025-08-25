
import os
import pandas as pd
import joblib
import urllib.request

# -------------------------------
# Helper to load local/download models
# -------------------------------
def load_or_download(path, url=None):
    """
    Load a joblib file. If not found locally, download it (if URL is given).
    """
    if not os.path.exists(path):
        if url:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            urllib.request.urlretrieve(url, path)
        else:
            raise FileNotFoundError(f"Missing required file: {path}")
    return joblib.load(path)


# -------------------------------
# Load models & scalers
# (replace URLs with cloud links if you don't push artifacts/ to GitHub)
# -------------------------------
model_young = load_or_download("artifacts/model_young.joblib")
model_rest = load_or_download("artifacts/model_rest.joblib")
scaler_young = load_or_download("artifacts/scaler_young.joblib")
scaler_rest = load_or_download("artifacts/scaler_rest.joblib")


# -------------------------------
# Risk score calculation
# -------------------------------
def calculate_normalized_risk(medical_history: str) -> float:
    risk_scores = {
        "diabetes": 6,
        "heart disease": 8,
        "high blood pressure": 6,
        "thyroid": 5,
        "no disease": 0,
        "none": 0,
    }

    diseases = medical_history.lower().split(" & ")
    total_risk_score = sum(risk_scores.get(disease.strip(), 0) for disease in diseases)

    max_score = 14  # heart disease (8) + diabetes/high BP (6)
    return total_risk_score / max_score


# -------------------------------
# Preprocessing input
# -------------------------------
def preprocess_input(input_dict: dict) -> pd.DataFrame:
    expected_columns = [
        "age", "number_of_dependants", "income_lakhs", "insurance_plan",
        "genetical_risk", "normalized_risk_score",
        "gender_Male", "region_Northwest", "region_Southeast", "region_Southwest",
        "marital_status_Unmarried", "bmi_category_Obesity", "bmi_category_Overweight",
        "bmi_category_Underweight", "smoking_status_Occasional", "smoking_status_Regular",
        "employment_status_Salaried", "employment_status_Self-Employed",
    ]

    insurance_plan_encoding = {"Bronze": 1, "Silver": 2, "Gold": 3}
    df = pd.DataFrame(0, columns=expected_columns, index=[0])

    # Fill based on user input
    for key, value in input_dict.items():
        if key == "Gender" and value == "Male":
            df["gender_Male"] = 1
        elif key == "Region":
            df[f"region_{value}"] = 1 if f"region_{value}" in df.columns else 0
        elif key == "Marital Status" and value == "Unmarried":
            df["marital_status_Unmarried"] = 1
        elif key == "BMI Category" and f"bmi_category_{value}" in df.columns:
            df[f"bmi_category_{value}"] = 1
        elif key == "Smoking Status" and f"smoking_status_{value}" in df.columns:
            df[f"smoking_status_{value}"] = 1
        elif key == "Employment Status" and f"employment_status_{value}" in df.columns:
            df[f"employment_status_{value}"] = 1
        elif key == "Insurance Plan":
            df["insurance_plan"] = insurance_plan_encoding.get(value, 1)
        elif key == "Age":
            df["age"] = value
        elif key == "Number of Dependants":
            df["number_of_dependants"] = value
        elif key == "Income in Lakhs":
            df["income_lakhs"] = value
        elif key == "Genetical Risk":
            df["genetical_risk"] = value

    # Add normalized risk
    df["normalized_risk_score"] = calculate_normalized_risk(input_dict["Medical History"])
    df = handle_scaling(input_dict["Age"], df)

    return df


# -------------------------------
# Handle scaling
# -------------------------------
def handle_scaling(age: int, df: pd.DataFrame) -> pd.DataFrame:
    scaler_object = scaler_young if age <= 25 else scaler_rest
    cols_to_scale = scaler_object["cols_to_scale"]
    scaler = scaler_object["scaler"]

    df["income_level"] = 0  # dummy col (scaler expects it)
    df[cols_to_scale] = scaler.transform(df[cols_to_scale])
    df.drop("income_level", axis="columns", inplace=True)
    return df


# -------------------------------
# Prediction function
# -------------------------------
def predict(input_dict: dict) -> int:
    input_df = preprocess_input(input_dict)
    model = model_young if input_dict["Age"] <= 25 else model_rest
    prediction = model.predict(input_df)
    return int(prediction[0])
