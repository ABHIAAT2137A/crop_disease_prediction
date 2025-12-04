# resave_model.py
import pickle
import os

SRC = "wheat_model.pkl"    # the existing pickle you have locally
DST = "wheat_model.pkl"    # we overwrite in place (or change DST if you prefer)

if not os.path.exists(SRC):
    raise FileNotFoundError(f"{SRC} not found. Put your existing pickle in repo root first.")

with open(SRC, "rb") as f:
    data = pickle.load(f)

# re-dump with protocol 4 for broad compatibility on Streamlit
with open(DST, "wb") as f:
    pickle.dump(data, f, protocol=4)

print("Re-saved pickle with protocol=4 to", DST)
