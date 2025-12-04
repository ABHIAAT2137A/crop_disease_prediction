import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import pickle
from datetime import datetime

# Load your dataset
df = pd.read_csv("wheat_crop_dataset_final_120_rows.csv")

# Reference date
REFERENCE_DATE = datetime(2025, 12, 2)

# Convert dates
df["sowing_date"] = pd.to_datetime(df["sowing_date"])
df["days_since_sowing"] = (REFERENCE_DATE - df["sowing_date"]).dt.days

# Encoders
stage_encoder = LabelEncoder()
disease_encoder = LabelEncoder()

df["stage_enc"] = stage_encoder.fit_transform(df["crop_stage"])
df["disease_enc"] = disease_encoder.fit_transform(df["crop_disease"])

# Features and target
X = df[["stage_enc", "days_since_sowing"]]
y = df["disease_enc"]

# Train model with sklearn 1.2.2
model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)
model.fit(X, y)

# Save model in Streamlit-compatible format
data = {
    "model": model,
    "stage_encoder": stage_encoder,
    "disease_encoder": disease_encoder
}

with open("wheat_model.pkl", "wb") as f:
    pickle.dump(data, f, protocol=4)

print("Streamlit-compatible wheat_model.pkl created successfully!")
