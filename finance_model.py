import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
import joblib
import os

# 📌 Load the dataset
csv_path = r"C:\Users\there\Documents\scripts\BI_finance\transactions.csv"
df = pd.read_csv(csv_path)

# 📌 1️⃣ Auto-label categories using keywords
category_rules = {
    "Netflix|Spotify|HBO|Disney|Apple Music|Prime Video": "Entertainment",
    "Uber|Taxi|Bus|Metro|Cabify|BlaBlaCar": "Transport",
    "IKEA|Zara|H&M|Amazon|AliExpress|Decathlon|Lidl": "Shopping",
    "Gym|Pharmacy|Doctor|Hospital|Dentist|Medicine|Wellness": "Health",
    "Salary|Payroll|Income|Bonus|Freelance|Invoice|Nómina": "Salary",
    "Transferencia|Ahorro|Saving|Deposit|Investment": "Savings",
    "Restaurant|McDonalds|KFC|Subway|Starbucks|Food Delivery": "Food & Dining",
    "Gas|Electricity|Water|Internet|Phone Bill": "Utilities",
    "Car|Mechanic|Insurance|Parking|Fuel|Gasoline|Toll": "Car Expenses"
}

def auto_label(concepto):
    concepto_lower = str(concepto).lower()
    for keywords, category in category_rules.items():
        if any(word in concepto_lower for word in keywords.lower().split("|")):
            return category
    return "Uncategorized"  # Default label if no match

# Apply auto-labeling
df["Category"] = df["Concepto"].apply(auto_label)

# 📌 2️⃣ Check category diversity before filtering
category_counts = df["Category"].value_counts()
print("Category distribution before filtering:")
print(category_counts)

# Ensure there are at least 2 unique categories for training
if category_counts.nunique() <= 1:
    print("⚠️ Warning: Low category variety. Consider improving category rules.")
else:
    print("✅ Sufficient category diversity detected.")

# 📌 3️⃣ Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(df["Concepto"], df["Category"], test_size=0.2, random_state=42)

# 📌 4️⃣ Train the model
model = make_pipeline(TfidfVectorizer(), MultinomialNB())
model.fit(X_train, y_train)

# 📌 5️⃣ Save the trained model
model_dir = r"C:\Users\there\Documents\scripts\models"
os.makedirs(model_dir, exist_ok=True)
model_path = r"C:\Users\there\Documents\scripts\BI_finance\models\expense_classifier.pkl"
joblib.dump(model, model_path)

print(f"✅ Model trained and saved to {model_path}!")
