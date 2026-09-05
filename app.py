import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

st.set_page_config(page_title="NBA Shot Quality & Zone Analysis", layout="wide")

@st.cache_data
def load_data():
    return pd.read_csv("shot_data_with_xfg.csv")

df = load_data()

st.title("NBA Expected Shot Quality (xFG%) & Court Heatmap Analysis")
st.write("Evaluate shot efficiency, court frequency, and area strengths/weaknesses across teams and players.")

#  SIDEBAR FILTERS 
st.sidebar.header("Filter Options")

# Filter 1: Select Team
teams = sorted(df['TEAM_NAME'].unique())
selected_team = st.sidebar.selectbox("Select Team:", teams)

# Filter dataset by selected team
team_df = df[df['TEAM_NAME'] == selected_team]

# Filter 2: Select Player (options updated based on team)
players = sorted(team_df['PLAYER_NAME'].unique())
selected_player = st.sidebar.selectbox("Select Player (or 'All Players'):", ["All Players"] + players)

# Apply player filter if selected
if selected_player != "All Players":
    filtered_df = team_df[team_df['PLAYER_NAME'] == selected_player]
    display_title = f"{selected_player} ({selected_team})"
else:
    filtered_df = team_df
    display_title = f"All Players - {selected_team}"

st.divider()

#  TOP LEVEL METRICS 
total_shots = len(filtered_df)
actual_fg = filtered_df['SHOT_MADE_FLAG'].mean() if total_shots > 0 else 0
expected_fg = filtered_df['EXPECTED_FG'].mean() if total_shots > 0 else 0
shoe = (actual_fg - expected_fg) * 100  # Shooting Over Expectation

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Shot Attempts", total_shots)
col2.metric("Actual FG%", f"{actual_fg * 100:.1f}%")
col3.metric("Expected FG% (xFG%)", f"{expected_fg * 100:.1f}%")
col4.metric("Shooting Over Expectation (ShoE)", f"{shoe:+.1f}%")

st.divider()

#  VISUALIZATIONS: SCATTER VS HEATMAP 
st.subheader(f"Shot Selection Analysis for {display_title}")

view_type = st.radio("Select View:", ["Scatter Plot (Makes/Misses)", "Density Heatmap (Volume Hotspots)"], horizontal=True)

fig, ax = plt.subplots(figsize=(8, 6))

if view_type == "Scatter Plot (Makes/Misses)":
    sns.scatterplot(
        data=filtered_df, 
        x='LOC_X', 
        y='LOC_Y', 
        hue='SHOT_MADE_FLAG', 
        palette={0: 'red', 1: 'green'},
        alpha=0.5,
        ax=ax
    )
    ax.set_title(f"Shot Chart: {display_title}")
else:
    # 2D Kernel Density Estimate (Heatmap)
    sns.kdeplot(
        data=filtered_df,
        x='LOC_X',
        y='LOC_Y',
        cmap="YlOrRd",
        fill=True,
        thresh=0.05,
        levels=10,
        ax=ax
    )
    ax.set_title(f"Court Volume Heatmap: {display_title}")

# Formatting half-court bounds
ax.set_xlabel("Court Location X")
ax.set_ylabel("Court Location Y")
ax.set_xlim(-250, 250)
ax.set_ylim(-50, 420)
st.pyplot(fig)

st.divider()

# STRENGTHS, WEAKNESSES & PLACES TO IMPROVE 
st.subheader("Zone Performance & Development Feedback")

# Group data by basic court zone
zone_summary = filtered_df.groupby('SHOT_ZONE_BASIC').agg(
    Attempts=('SHOT_MADE_FLAG', 'count'),
    Actual_FG=('SHOT_MADE_FLAG', 'mean'),
    Expected_FG=('EXPECTED_FG', 'mean')
).reset_index()

zone_summary['ShoE_Zone'] = (zone_summary['Actual_FG'] - zone_summary['Expected_FG']) * 100

# Identify Key Zones
if not zone_summary.empty:
    best_zone = zone_summary.sort_values(by='ShoE_Zone', ascending=False).iloc[0]
    worst_zone = zone_summary.sort_values(by='ShoE_Zone', ascending=True).iloc[0]
    most_frequent_zone = zone_summary.sort_values(by='Attempts', ascending=False).iloc[0]

    col_a, col_b, col_c = st.columns(3)
    
    with col_a:
        st.success(f"**Primary Strength Area**\n\n"
                   f"**Zone:** {best_zone['SHOT_ZONE_BASIC']}\n\n"
                   f"**ShoE:** {best_zone['ShoE_Zone']:+.1f}%\n\n"
                   f"*Outperforming shot difficulty in this area.*")
        
    with col_b:
        st.error(f" **Area Needing Improvement**\n\n"
                 f"**Zone:** {worst_zone['SHOT_ZONE_BASIC']}\n\n"
                 f"**ShoE:** {worst_zone['ShoE_Zone']:+.1f}%\n\n"
                 f"*Shooting below expected model probability here.*")
        
    with col_c:
        st.info(f"**Highest Volume Spot**\n\n"
                f"**Zone:** {most_frequent_zone['SHOT_ZONE_BASIC']}\n\n"
                f"**Attempts:** {most_frequent_zone['Attempts']} shots\n\n"
                f"*Represents the primary tendency on the court.*")

# Detailed Breakdown Table
st.write("### Detailed Zone Breakdown")
display_table = zone_summary.copy()
display_table['Actual_FG'] = (display_table['Actual_FG'] * 100).round(1).astype(str) + '%'
display_table['Expected_FG'] = (display_table['Expected_FG'] * 100).round(1).astype(str) + '%'
display_table['ShoE_Zone'] = display_table['ShoE_Zone'].round(1).astype(str) + '%'

st.dataframe(
    display_table.rename(columns={
        'SHOT_ZONE_BASIC': 'Court Zone',
        'Attempts': 'Shot Volume',
        'Actual_FG': 'Actual FG%',
        'Expected_FG': 'Expected FG%',
        'ShoE_Zone': 'Shooting vs Expectation'
    }), 
    use_container_width=True
)