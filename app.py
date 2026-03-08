import streamlit as st
import pandas as pd
import numpy as np
import os
import plotly.express as px
import plotly.graph_objects as go

# --- 1. PAGE SETUP ---
st.set_page_config(
    page_title="ATM Bottleneck Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- HIDE STREAMLIT BRANDING (Professional Look) ---
hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)

# --- 2. DATA LOADER ---
@st.cache_data
def load_data():
    csv_file = "atm_logs.csv"
    
    # Check if your YOLO script has generated the file yet
    if os.path.exists(csv_file):
        df = pd.read_csv(csv_file)
        if not df.empty:
            return df, True # True means real data

    # FALLBACK: Generate mock data so you can still demo it to the client
    np.random.seed(42)
    data = []
    for i in range(100):
        times = [
            np.random.randint(2, 6),    # Approach
            np.random.randint(3, 8),    # Card Insert
            np.random.randint(15, 45),  # PIN Entry (Simulated Bottleneck)
            np.random.randint(2, 5),    # Card Retrieve
            np.random.randint(3, 10)    # Cash Out
        ]
        total = sum(times)
        success = np.random.choice([True, False], p=[0.90, 0.10])
        data.append([f"TXN-{1000+i}", total, times[0], times[1], times[2], times[3], times[4], success])
    
    df = pd.DataFrame(data, columns=["Transaction_ID", "Total_Duration_Sec", "Approach_Sec", 
                                     "Card_Insert_Sec", "PIN_Entry_Sec", "Card_Retrieve_Sec", 
                                     "Cash_Out_Sec", "Success"])
    return df, False # False means mock data

df, is_real_data = load_data()

# --- 3. HEADER ---
st.title("🏦 ATM Interaction & Bottleneck Analytics")
if is_real_data:
    st.success("🟢 Live Data Connected: Reading from atm_logs.csv")
else:
    st.warning("🟡 Demo Mode: 'atm_logs.csv' not found. Showing simulated data for client presentation.")
st.divider()

# --- 4. TOP KPI METRICS ---
col1, col2, col3, col4 = st.columns(4)

avg_total_time = df["Total_Duration_Sec"].mean()
success_rate = (df["Success"].sum() / len(df)) * 100
avg_pin_time = df["PIN_Entry_Sec"].mean()

with col1:
    st.metric("Total Transactions", len(df))
with col2:
    st.metric("Avg Total Time", f"{avg_total_time:.1f}s")
with col3:
    st.metric("Transaction Success Rate", f"{success_rate:.1f}%")
with col4:
    st.metric("⚠️ Longest Phase (PIN/Keypad)", f"{avg_pin_time:.1f}s", delta="Bottleneck Detected", delta_color="inverse")

st.markdown("<br>", unsafe_allow_html=True)

# --- 5. INTERACTIVE CHARTS ---
col_left, col_right = st.columns(2)

with col_left:
    st.markdown("### ⏳ Average Time Spent per Phase")
    # Grab the 4 main ATM phases from the CSV
    phase_cols = ["Card_Insert_Sec", "PIN_Entry_Sec", "Card_Retrieve_Sec", "Cash_Out_Sec"]
    avg_times = df[phase_cols].mean().reset_index()
    avg_times.columns = ["Phase", "Average Time (Seconds)"]
    
    # Clean up the names for the chart
    avg_times["Phase"] = ["1. Insert Card", "2. Use Keypad", "3. Take Card", "4. Take Cash"]
    
    fig_bar = px.bar(avg_times, x="Phase", y="Average Time (Seconds)", 
                     text_auto='.1f', 
                     color="Phase",
                     color_discrete_sequence=px.colors.sequential.Teal)
    fig_bar.update_layout(showlegend=False, plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig_bar, use_container_width=True)

with col_right:
    st.markdown("### 🔻 User Journey Funnel")
    
    # Calculate how many people made it through each phase
    started = len(df)
    inserted = len(df[df["Card_Insert_Sec"] > 0])
    pinned = len(df[df["PIN_Entry_Sec"] > 0])
    card_taken = len(df[df["Card_Retrieve_Sec"] > 0])
    cashed_out = df["Success"].sum()

    funnel_data = dict(
        number=[started, inserted, pinned, card_taken, cashed_out],
        stage=["Approached", "Inserted Card", "Used Keypad", "Took Card", "Took Cash"]
    )
    
    fig_funnel = go.Figure(go.Funnel(
        y = funnel_data["stage"],
        x = funnel_data["number"],
        textinfo = "value+percent initial",
        marker = {"color": ["#1f77b4", "#1f77b4", "#ff7f0e", "#1f77b4", "#1f77b4"]} # Orange marks the bottleneck
    ))
    fig_funnel.update_layout(plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig_funnel, use_container_width=True)

st.divider()

# --- 6. RAW LOGS ---
st.markdown("### 📋 Raw Computer Vision Logs")
st.dataframe(df.tail(15).iloc[::-1], use_container_width=True) # Shows the 15 most recent transactions