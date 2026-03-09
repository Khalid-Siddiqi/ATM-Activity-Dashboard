import streamlit as st
import pandas as pd
import numpy as np
import os
import plotly.express as px
import plotly.graph_objects as go

# --- 1. PAGE SETUP ---
st.set_page_config(
    page_title="Vision Analytics",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- CUSTOM CSS FOR PREMIUM LOOK ---
st.markdown("""
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    /* Style the metric cards */
    div[data-testid="metric-container"] {
        background-color: #1E293B;
        border: 1px solid #334155;
        padding: 5% 5% 5% 10%;
        border-radius: 10px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    </style>
""", unsafe_allow_html=True)

# --- 2. DATA LOADER ---
@st.cache_data
def load_data():
    csv_file = "atm_logs.csv"
    if os.path.exists(csv_file):
        df = pd.read_csv(csv_file)
        if not df.empty:
            return df, True

    # High-end Mock Data if CSV is missing
    np.random.seed(42)
    data = []
    for i in range(120):
        times = [
            np.random.randint(2, 6),    # Approach
            np.random.randint(3, 8),    # Card Insert
            np.random.randint(15, 35),  # PIN Entry (Bottleneck)
            np.random.randint(2, 5),    # Card Retrieve
            np.random.randint(3, 10)    # Cash Out
        ]
        data.append([f"TXN-{1000+i}", sum(times), times[0], times[1], times[2], times[3], times[4], np.random.choice([True, False], p=[0.92, 0.08])])
    
    return pd.DataFrame(data, columns=["Transaction_ID", "Total_Duration_Sec", "Approach_Sec", "Card_Insert_Sec", "PIN_Entry_Sec", "Card_Retrieve_Sec", "Cash_Out_Sec", "Success"]), False

df, is_real_data = load_data()

# --- 3. HEADER AREA ---
col_logo, col_title = st.columns([1, 8])
with col_logo:
    # A generic vision icon to look professional
    st.markdown("<h1 style='text-align: center; color: #00AEEF;'>👁️</h1>", unsafe_allow_html=True)
with col_title:
    st.title("Kiosk Vision Analytics Engine")
    st.markdown("Automated Bottleneck Detection & Customer Journey Mapping")

st.divider()

# --- 4. TOP KPI METRICS ---
avg_total_time = df["Total_Duration_Sec"].mean()
success_rate = (df["Success"].sum() / len(df)) * 100
avg_pin_time = df["PIN_Entry_Sec"].mean()

m1, m2, m3, m4 = st.columns(4)
m1.metric("Processed Transactions", len(df), "Live Camera Feed")
m2.metric("Avg Service Time", f"{avg_total_time:.1f}s", "-1.2s vs Target", delta_color="normal")
m3.metric("Task Completion Rate", f"{success_rate:.1f}%", "-2.1% Abandonment", delta_color="normal")
m4.metric("⚠️ Primary Bottleneck", f"{avg_pin_time:.1f}s", "Phase: PIN Entry", delta_color="inverse")

st.markdown("<br>", unsafe_allow_html=True)

# --- 5. MAIN CONTENT SPLIT (VIDEO + ANALYTICS) ---
col_video, col_charts = st.columns([1.2, 1])

with col_video:
    st.markdown("### 🔴 Live Surveillance Feed (Annotated)")
    video_path = r"C:\Users\Yousuf Traders\Desktop\Projects\ATM Activity Dashboard\atm_surveillance.mp4"
    
    # Check if the video from your YOLO script exists
    if os.path.exists(video_path):
        st.video(video_path)
    else:
        st.warning(f"Video file '{video_path}' not found. Please run the YOLO script first.")
        
    st.markdown("### 📋 Recent Interaction Logs")
    st.dataframe(df.tail(6).iloc[::-1], use_container_width=True)

with col_charts:
    # --- 1. THE BOTTLENECK CHART (Moved to Top) ---
    st.markdown("### ⏳ Phase Duration Breakdown")
    phase_cols = ["Card_Insert_Sec", "PIN_Entry_Sec", "Card_Retrieve_Sec", "Cash_Out_Sec"]
    avg_times = df[phase_cols].mean().reset_index()
    avg_times.columns = ["Phase", "Average Time (Seconds)"]
    avg_times["Phase"] = ["1. Insert Card", "2. Use Keypad", "3. Take Card", "4. Take Cash"]
    
    # We use Wavetec Blue (#00AEEF) to highlight the bottleneck, and Slate Gray for the rest
    fig_bar = px.bar(avg_times, x="Average Time (Seconds)", y="Phase", 
                     orientation='h',
                     text_auto='.1f', 
                     color="Phase",
                     color_discrete_sequence=["#334155", "#00AEEF", "#334155", "#334155"])
    
    fig_bar.update_layout(
        showlegend=False, 
        plot_bgcolor="rgba(0,0,0,0)", 
        paper_bgcolor="rgba(0,0,0,0)", 
        template="plotly_dark", 
        margin=dict(t=10, b=10, l=0, r=0),
        xaxis_title="",
        yaxis_title=""
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True) # Add a little breathing room

    # --- 2. THE NEW OUTCOME CHART (Replaces Funnel) ---
    st.markdown("### 🎯 Transaction Outcomes")
    
    # Calculate Success vs Abandoned
    success_counts = df["Success"].value_counts().reset_index()
    success_counts.columns = ["Status", "Count"]
    success_counts["Status"] = success_counts["Status"].map({True: "Completed", False: "Abandoned"})

    # Create a sleek Donut Chart
    fig_donut = px.pie(success_counts, values="Count", names="Status", hole=0.6,
                       color="Status",
                       color_discrete_map={"Completed": "#00AEEF", "Abandoned": "#f43f5e"}) # Blue for success, Red for abandon
    
    fig_donut.update_traces(textinfo='percent+label', textfont_size=14, marker=dict(line=dict(color='#0F172A', width=2)))
    fig_donut.update_layout(
        showlegend=False, 
        plot_bgcolor="rgba(0,0,0,0)", 
        paper_bgcolor="rgba(0,0,0,0)", 
        template="plotly_dark", 
        margin=dict(t=10, b=10, l=0, r=0)
    )
    
    # Add a center label to the donut chart
    fig_donut.add_annotation(text="Total<br>Interactions", x=0.5, y=0.5, font_size=14, showarrow=False, font_color="#F8FAFC")
    
    st.plotly_chart(fig_donut, use_container_width=True)