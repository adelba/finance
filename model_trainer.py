import joblib
import os
import pandas as pd

# Load the trained model
model_path = os.path.join(os.getcwd(), "models", "expense_classifier.pkl")
if not os.path.exists(model_path):
    raise FileNotFoundError("❌ Model not found! Train the model first using train_model.py")

model = joblib.load(model_path)

# Load new transactions from CSV
new_transactions_path = r"C:\Users\there\Documents\scripts\BI_finance\transactions_test.csv"

if not os.path.exists(new_transactions_path):
    raise FileNotFoundError(f"❌ CSV file not found: {new_transactions_path}")

df_new = pd.read_csv(new_transactions_path)

# Ensure the required column exists
if "Concepto" not in df_new.columns:
    raise ValueError("❌ CSV must contain a 'Concepto' column!")

# Predict categories
df_new["Predicted Category"] = model.predict(df_new["Concepto"])

# Save results to a new CSV
output_path = r"C:\Users\there\Documents\scripts\BI_finance\new_transactions_classified.csv"
df_new.to_csv(output_path, index=False)

print(f"✅ Predictions saved to {output_path}!")
