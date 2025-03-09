import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.linear_model import LinearRegression
from fpdf import FPDF
import base64

# Load Data
st.sidebar.header("Upload CSV File")
uploaded_file = st.sidebar.file_uploader("Upload your finance data (CSV format)", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # Ensure Fecha valor is a datetime column
    df['Fecha valor'] = pd.to_datetime(df['Fecha valor'], dayfirst=True)

    # Ensure YearMonth column exists
    df['YearMonth'] = df['Fecha valor'].dt.to_period('M').astype(str)

    # Ensure Income and Expense columns exist
    if 'Income' not in df.columns:
        df['Income'] = df['Importe'].apply(lambda x: x if x > 0 else 0)
    if 'Expense' not in df.columns:
        df['Expense'] = df['Importe'].apply(lambda x: abs(x) if x < 0 else 0)

    # Debugging step
    st.write("Columns in uploaded file:", df.columns.tolist())
    st.write(df.head())  # Ensure YearMonth exists

    # Group by YearMonth
    monthly_summary = df.groupby("YearMonth")[["Income", "Expense"]].sum()

    # Display results
    st.write("### Monthly Income vs Expenses")
    st.dataframe(monthly_summary)

else:
    st.warning("⚠️ Please upload a CSV file to proceed.")

# ---- HEADER ----
st.title("📊 Finance Dashboard")

# ---- RAW DATA ----
st.header("Raw Data")
st.dataframe(df.head(10))

# ---- KEY FINANCIAL METRICS ----
total_income = df['Income'].sum()
total_expense = df['Expense'].sum()
net_savings = total_income - total_expense
savings_rate = (net_savings / total_income) * 100 if total_income > 0 else 0

st.header("📈 Key Financial Metrics")
st.markdown(f"""
- 💰 **Total Income:** `{total_income:,.2f} €`
- 💸 **Total Expenses:** `{total_expense:,.2f} €`
- 🏦 **Net Savings:** `{net_savings:,.2f} €`
- 📉 **Savings Rate:** `{savings_rate:.2f} %`
""")

# ---- EXPENSES BY CATEGORY ----
st.header("🛒 Expense Analysis by Category")

if "Predicted Category" in df.columns:
    category_expense = df.groupby("Predicted Category")["Expense"].sum().sort_values(ascending=False)
    
    fig, ax = plt.subplots(figsize=(8, 5))
    category_expense[:5].plot(kind='pie', autopct='%1.1f%%', cmap='coolwarm', ax=ax)
    ax.set_title("Top 5 Expense Categories")
    ax.set_ylabel("")
    st.pyplot(fig)
else:
    st.warning("No 'Category' column found in dataset.")

# ---- INCOME VS EXPENSES (BAR CHART) ----
st.header("📊 Monthly Income vs Expenses")

monthly_summary = df.groupby("YearMonth")[["Income", "Expense"]].sum()

fig, ax = plt.subplots(figsize=(8, 5))
bar_width = 0.4
index = np.arange(len(monthly_summary))

ax.bar(index, monthly_summary["Income"], width=bar_width, label="Income", color="green")
ax.bar(index + bar_width, monthly_summary["Expense"], width=bar_width, label="Expense", color="red")

ax.set_xlabel("Month")
ax.set_ylabel("Amount (€)")
ax.set_title("Monthly Income vs Expenses")
ax.set_xticks(index + bar_width / 2)
ax.set_xticklabels(monthly_summary.index, rotation=45)
ax.legend()

st.pyplot(fig)

# ---- BUDGET TRACKER ----
st.sidebar.header("💰 Set Your Monthly Budget")
monthly_budget = st.sidebar.number_input("Enter your budget (€)", min_value=0, value=1000)

selected_month = df["YearMonth"].max()
selected_month_expense = df[df["YearMonth"] == selected_month]["Expense"].sum()

st.header("🎯 Budget Tracker")
progress = min(1.0, selected_month_expense / monthly_budget)
st.progress(progress)

st.markdown(f"""
- **💸 Spent:** `{selected_month_expense:,.2f} €`
- **💵 Remaining:** `{monthly_budget - selected_month_expense:,.2f} €`
""")

# ---- INCOME & EXPENSE FORECASTING ----
st.header("🔮 Next Month Predictions")

df["MonthIndex"] = np.arange(len(df))
X = df[["MonthIndex"]]
y_income = df["Income"]
y_expense = df["Expense"]

model_income = LinearRegression().fit(X, y_income)
model_expense = LinearRegression().fit(X, y_expense)

next_month_index = df["MonthIndex"].max() + 1
predicted_income = model_income.predict([[next_month_index]])[0]
predicted_expense = model_expense.predict([[next_month_index]])[0]

st.markdown(f"""
- **📈 Predicted Income:** `{predicted_income:,.2f} €`
- **📉 Predicted Expenses:** `{predicted_expense:,.2f} €`
""")

# ---- DOWNLOADABLE PDF REPORT ----
import base64

def generate_pdf():
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    def safe_text(text):
        """Ensure text is in a format FPDF can handle by encoding & decoding it safely."""
        return text.encode("latin-1", "replace").decode("latin-1")

    pdf.cell(200, 10, safe_text("Finance Report"), ln=True, align='C')
    pdf.ln(10)

    pdf.cell(200, 10, safe_text(f"Total Income: {total_income:.2f} €"), ln=True)
    pdf.cell(200, 10, safe_text(f"Total Expenses: {total_expense:.2f} €"), ln=True)
    pdf.cell(200, 10, safe_text(f"Net Savings: {net_savings:.2f} €"), ln=True)
    pdf.cell(200, 10, safe_text(f"Savings Rate: {savings_rate:.2f} %"), ln=True)

    pdf_path = "Finance_Report.pdf"
    pdf.output(pdf_path)

    with open(pdf_path, "rb") as f:
        pdf_bytes = f.read()
        b64 = base64.b64encode(pdf_bytes).decode()

    href = f'<a href="data:application/octet-stream;base64,{b64}" download="Finance_Report.pdf">📥 Click here to download the report</a>'
    
    # Store in session state so button can trigger the display
    st.session_state["download_link"] = href

# ✅ Make sure button displays the link inside the sidebar
if st.sidebar.button("📥 Generate & Download Financial Report"):
    generate_pdf()

# ✅ Show download link *only* after clicking button
if "download_link" in st.session_state:
    st.sidebar.markdown(st.session_state["download_link"], unsafe_allow_html=True)
# ---- END ----
st.markdown("---")
st.caption("📌 Developed by the one and only Odelov with ❤️ using Streamlit")
