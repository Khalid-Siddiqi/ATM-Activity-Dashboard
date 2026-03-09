import streamlit as st
import pandas as pd
import numpy as np
import os
import plotly.express as px
import plotly.graph_objects as go

# --- 1. PAGE SETUP ---
st.set_page_config(
    page_title="Wavetec | Vision Analytics",
    page_icon="📈", 
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- CUSTOM CSS FOR EXECUTIVE BOARDROOM LOOK ---
st.markdown("""
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Sleek, corporate styling for the KPI cards */
    div[data-testid="metric-container"] {
        background-color: #121E36;
        border: 1px solid #1E325C;
        border-left: 4px solid #00AEEF; /* Wavetec Cyan Accent */
        padding: 5% 5% 5% 10%;
        border-radius: 6px;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.2);
    }
    
    /* Clean up the headers */
    h1, h2, h3 {
        font-weight: 300 !important;
        letter-spacing: 0.5px;
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

    # High-end Mock Data
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

# --- 3. HEADER AREA (CLEAN & PROFESSIONAL) ---
st.markdown("<h1 style='color: #F8FAFC;'>Wavetec <span style='color: #00AEEF; font-weight: 600;'>Vision Analytics</span></h1>", unsafe_allow_html=True)
st.markdown("<p style='color: #94A3B8; font-size: 1.1rem;'>Automated Customer Flow Mapping & Journey Optimization</p>", unsafe_allow_html=True)
st.divider()

# --- 4. TOP KPI METRICS ---
avg_total_time = df["Total_Duration_Sec"].mean()
success_rate = (df["Success"].sum() / len(df)) * 100
avg_pin_time = df["PIN_Entry_Sec"].mean()

m1, m2, m3, m4 = st.columns(4)
m1.metric("Total Interactions Mapped", len(df), "Active Edge Node")
m2.metric("Avg Customer Journey", f"{avg_total_time:.1f}s", "-1.2s vs Target", delta_color="normal")
m3.metric("Journey Completion Rate", f"{success_rate:.1f}%", "-2.1% Abandonment", delta_color="normal")
m4.metric("Critical Friction Point", f"{avg_pin_time:.1f}s", "Phase: PIN Entry", delta_color="inverse")

st.markdown("<br>", unsafe_allow_html=True)

# --- 5. MAIN CONTENT SPLIT ---
col_video, col_charts = st.columns([1.2, 1])

with col_video:
    st.markdown("### 📷 Edge Device Feed (Anonymized Tracking)")
    # Changed wording from "Surveillance" to "Edge Device Feed" - sounds much more enterprise-tech.
    video_path = "processed_whatsapp_10.mp4" # Update this to one of your actual output videos
    
    if os.path.exists(video_path):
        st.video(video_path)
    else:
        st.info("Video feed standing by. Please ensure the pipeline has processed the latest batch.")
        
    st.markdown("### 📋 Spatial Interaction Logs")
    st.dataframe(df.tail(6).iloc[::-1], use_container_width=True)

with col_charts:
    # --- PHASE DURATION CHART ---
    st.markdown("### ⏳ Journey Phase Breakdown")
    phase_cols = ["Card_Insert_Sec", "PIN_Entry_Sec", "Card_Retrieve_Sec", "Cash_Out_Sec"]
    avg_times = df[phase_cols].mean().reset_index()
    avg_times.columns = ["Phase", "Average Time (Seconds)"]
    avg_times["Phase"] = ["1. Insert Card", "2. Use Keypad", "3. Take Card", "4. Take Cash"]
    
    # Using deep corporate blue for standard, Cyan for the bottleneck
    fig_bar = px.bar(avg_times, x="Average Time (Seconds)", y="Phase", 
                     orientation='h',
                     text_auto='.1f', 
                     color="Phase",
                     color_discrete_sequence=["#1E325C", "#00AEEF", "#1E325C", "#1E325C"])
    
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

    st.markdown("<br>", unsafe_allow_html=True) 

    # --- TRANSACTION OUTCOMES DONUT ---
    st.markdown("### 🎯 Journey Outcomes")
    
    success_counts = df["Success"].value_counts().reset_index()
    success_counts.columns = ["Status", "Count"]
    success_counts["Status"] = success_counts["Status"].map({True: "Completed", False: "Abandoned"})

    # Clean, executive donut chart
    fig_donut = px.pie(success_counts, values="Count", names="Status", hole=0.6,
                       color="Status",
                       color_discrete_map={"Completed": "#00AEEF", "Abandoned": "#FF4B4B"}) 
    
    fig_donut.update_traces(textinfo='percent+label', textfont_size=14, marker=dict(line=dict(color='#0A1128', width=3)))
    fig_donut.update_layout(
        showlegend=False, 
        plot_bgcolor="rgba(0,0,0,0)", 
        paper_bgcolor="rgba(0,0,0,0)", 
        template="plotly_dark", 
        margin=dict(t=10, b=10, l=0, r=0)
    )
    
    fig_donut.add_annotation(text="Total<br>Interactions", x=0.5, y=0.5, font_size=16, showarrow=False, font_color="#F8FAFC")
    st.plotly_chart(fig_donut, use_container_width=True)