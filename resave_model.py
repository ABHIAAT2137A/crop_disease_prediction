import pickle
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

# Load your CSV again and retrain the model correctly

df = pd.read_csv("wheat_crop_dataset_final_120_rows.csv")

stage_encoder = LabelEncoder()
disease_encoder = LabelEncoder()

df["crop_stage"] = stage_encoder.fit_transform(df["crop_stage"])
df["crop_disease"] = disease_encoder.fit_transform(df["crop_disease"])

X = df[["crop_stage", "days_since_sowing"]]
y = df["crop_disease"]

model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X, y)

# Save with protocol=4 (maximum compatibility)
data = {
    "model": model,
    "stage_encoder": stage_encoder,
    "disease_encoder": disease_encoder
}

with open("wheat_model.pkl", "wb") as f:
    pickle.dump(data, f, protocol=4)

print("New wheat_model.pkl saved successfully!")
