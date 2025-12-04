# train.py
import pandas as pd
from datetime import datetime
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle
import os

CSV = "wheat_crop_dataset_final_120_rows.csv"  # ensure this CSV is in repo root or update path
OUT_PKL = "wheat_model.pkl"

if not os.path.exists(CSV):
    raise FileNotFoundError(f"{CSV} not found in repo root. Place your dataset there or update path in train.py")

df = pd.read_csv(CSV)

# ensure columns exist
required = ["crop_stage", "days_since_sowing", "crop_disease"]
if not all(c in df.columns for c in required):
    # if your CSV uses sowing_date instead of days_since_sowing, compute days
    if "sowing_date" in df.columns:
        REF_DATE = datetime(2025, 12, 2)
        df["sowing_date"] = pd.to_datetime(df["sowing_date"], errors="coerce")
        df = df.dropna(subset=["sowing_date"])
        df["days_since_sowing"] = (REF_DATE - df["sowing_date"]).dt.days
    else:
        raise KeyError(f"CSV missing required columns. Found: {df.columns.tolist()}")

# keep only needed columns
df = df.dropna(subset=["crop_stage", "days_since_sowing", "crop_disease"])
df = df[["crop_stage", "days_since_sowing", "crop_disease"]]

# encode
stage_encoder = LabelEncoder()
disease_encoder = LabelEncoder()

df["crop_stage_enc"] = stage_encoder.fit_transform(df["crop_stage"])
df["disease_enc"] = disease_encoder.fit_transform(df["crop_disease"])

X = df[["crop_stage_enc", "days_since_sowing"]]
y = df["disease_enc"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.18, random_state=42, stratify=y)

model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

acc = model.score(X_test, y_test)
print(f"Test accuracy: {acc*100:.2f}%")

# Save model and encoders together as a dict (protocol 4)
out = {
    "model": model,
    "stage_encoder": stage_encoder,
    "disease_encoder": disease_encoder
}
with open(OUT_PKL, "wb") as f:
    pickle.dump(out, f, protocol=4)

print(f"Saved {OUT_PKL}")
