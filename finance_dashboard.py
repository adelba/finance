import streamlit as st 
import pandas as pd
import matplotlib.pyplot as plt

def load_data(uploaded_file):
    df = pd.read_csv(uploaded_file)
    df['Fecha valor'] = pd.to_datetime(df['Fecha valor'], dayfirst=True)
    df['Month'] = df['Fecha valor'].dt.to_period('M')
    df = df.sort_values(by=['Fecha valor'])
    df = df.groupby('Fecha valor').last().reset_index()
    return df

def plot_trend(df, selected_month, title):
    if selected_month != "All Time":
        df = df[df['Month'] == selected_month]
    
    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.plot(df['Fecha valor'], df['Saldo'], marker='o', linestyle='-', label='Saldo', color='blue')
    ax1.set_xlabel("Fecha valor")
    ax1.set_ylabel("Saldo (€)", color='blue')  # Label with € symbol
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1.grid()
    
    ax2 = ax1.twinx()
    colors = df['Importe'].apply(lambda x: 'red' if x < 0 else 'green')
    ax2.bar(df['Fecha valor'], df['Importe'].abs(), color=colors, alpha=0.5, label='Importe')
    ax2.set_ylabel("Importe (€)", color='green')  # Label with € symbol
    ax2.tick_params(axis='y', labelcolor='green')
    
    plt.title(title)
    fig.tight_layout()
    st.pyplot(fig)

def main():
    st.title("Finance Dashboard")
    st.sidebar.header("Upload CSV File")
    uploaded_file = st.sidebar.file_uploader("Upload your finance data (CSV format)", type=["csv"])
    
    if uploaded_file:
        df = load_data(uploaded_file)
        st.write("## Raw Data")
        st.dataframe(df)
        
        # Calculate statistics for 'Saldo'
        max_saldo = df['Saldo'].max()
        min_saldo = df['Saldo'].min()
        
        # Exclude max and min for mean calculation
        df_filtered = df[(df['Saldo'] != max_saldo) & (df['Saldo'] != min_saldo)]
        mean_saldo = df_filtered['Saldo'].mean()
        
        # Display max, min, and mean values with € symbol
        st.write("### Saldo Statistics")
        st.write(f"Maximum Saldo: {max_saldo:,.2f} €")
        st.write(f"Minimum Saldo: {min_saldo:,.2f} €")
        st.write(f"Mean Saldo (excluding max and min): {mean_saldo:,.2f} €")
        
        months = df['Month'].astype(str).unique().tolist()
        months.insert(0, "All Time")
        selected_month = st.sidebar.selectbox("Select a Month to analyze", months)
        
        plot_trend(df, selected_month, f"Trend for {selected_month}")

if __name__ == "__main__":
    main()
