import streamlit as st
import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model
import plotly.graph_objects as go

model = load_model("best_model.h5")
encoder = joblib.load("team_encoder.pkl")
scaler = joblib.load("scaler.pkl")

drive_df = pd.read_csv("drive_data.csv")

team_colors = {
    'LAC': '#0073CF', 'ARI': '#97233F', 'MIN': '#4F2683', 'CHI': '#0B162A',
    'DEN': '#FB4F14', 'DET': '#0076B6', 'SEA': '#002244', 'CLE': '#311D00',
    'GB': '#203731', 'PHI': '#004C54', 'DAL': '#041E42', 'MIA': '#008E97',
    'SF': '#AA0000', 'TB': '#D50A0A', 'BUF': '#00338D', 'BAL': '#241773',
    'CAR': '#0085CA', 'LV': '#A5ACAF', 'ATL': '#A71930', 'TEN': '#4B92DB',
    'IND': '#002C5F', 'KC': '#E31837', 'NYJ': '#125740', 'CIN': '#FB4F14',
    'PIT': '#FFB612', 'NO': '#D3BC8D', 'NE': '#002244', 'NYG': '#0B2265',
    'WAS': '#773141', 'LA': '#003594', 'JAX': '#006778', 'NA': '#000000'
}

teams = [
    "BUF", "NE", "MIA", "NYJ", "PIT", "BAL", "CIN", "CLE", "TEN", "IND", "HOU", "JAX",
    "KC", "DEN", "LAC", "LV", "DAL", "PHI", "WAS", "NYG", "DET", "GB", "CHI", "MIN", 
    "CAR", "TB", "NO", "ATL", "SEA", "ARI", "SF", "LAR"
]

st.set_page_config(page_title="NFL Drive Prediction", layout="wide")
st.title("NFL Drive Prediction")

st.sidebar.header("Enter Drive Information")
team = st.sidebar.selectbox("Select Team", teams)
starting_yardline = st.sidebar.number_input("Starting Yardline (0 to 100)", min_value=0, max_value=100, value=50)
num_plays = st.sidebar.number_input("Number of Plays", min_value=1, value=10)
drive_time_of_possession = st.sidebar.text_input("Drive Time of Possession (MM:SS)", "05:00")
first_downs = st.sidebar.number_input("First Downs", min_value=0, value=2)
score_differential = st.sidebar.number_input("Score Differential", min_value=-50, max_value=50, value=0)
timeouts = st.sidebar.number_input("Remaining Timeouts", min_value=0, max_value=3, value=3)
season = st.sidebar.number_input("Season", min_value = 2000, max_value = 2024, value =2023)

def time_to_seconds(time_str):
    try:
        minutes, seconds = time_str.split(':')
        return int(minutes) * 60 + int(seconds)
    except Exception as e:
        return np.nan

drive_time_of_possession_seconds = time_to_seconds(drive_time_of_possession)

X_input = pd.DataFrame({
    'num_plays': [num_plays],
    'starting_yardline': [starting_yardline],
    'drive_time_of_possession_seconds': [drive_time_of_possession_seconds],
    'first_downs': [first_downs],
    'score_differential': [score_differential],
    'team_encoded': encoder.transform([team]),
    'season': season,
    'timeouts': [timeouts],
})

X_input_scaled = scaler.transform(X_input)

prediction = model.predict(X_input_scaled)
predicted_class = np.argmax(prediction, axis=1)

points_mapping = {0: "No Points", 1: "Field Goal", 2: "Touchdown"}
predicted_points = points_mapping[predicted_class[0]]

tab = st.radio("Choose a tab", ("Drive Prediction", "Graph"))

if tab == "Drive Prediction":
    st.subheader(f"Predicted Drive Outcome for {team}")
    st.write(f"**Predicted Outcome:** {predicted_points}")

if tab == "Graph":
    actual_points = drive_df.groupby(['team', 'season'])['drive_points'].mean().reset_index()

    X_pred = drive_df[['num_plays', 'starting_yardline', 'drive_time_of_possession_seconds',
                       'first_downs', 'score_differential', 'team_encoded', 'season', 'timeouts']]
    X_pred = scaler.fit_transform(X_pred)
    predicted_probs = model.predict(X_pred)
    predicted_points = np.argmax(predicted_probs, axis=1) * 3 

    drive_df['predicted_drive_points'] = predicted_points

    predicted_points_avg = drive_df.groupby(['team', 'season'])['predicted_drive_points'].mean().reset_index()

    comparison_df = actual_points.merge(predicted_points_avg, on=['team', 'season'], suffixes=('_actual', '_predicted'))

    fig = go.Figure()
    
    for team in comparison_df['team'].unique():
        team_data = comparison_df[comparison_df['team'] == team]
        
        fig.add_trace(go.Scatter(
            x=team_data['season'], 
            y=team_data['drive_points'],
            mode='lines+markers', 
            name=f'Actual - {team}',
            line=dict(color=team_colors.get(team, '#636EFA'), width=4),
            marker=dict(size=6),
            legendgroup=team,
            visible=True if team == comparison_df['team'].unique()[0] else 'legendonly'
        ))
        
        fig.add_trace(go.Scatter(
            x=team_data['season'], 
            y=team_data['predicted_drive_points'],
            mode='lines+markers', 
            name=f'Predicted - {team}',
            line=dict(dash='dot', color=team_colors.get(team, '#636EFA'), width=4),
            marker=dict(size=6, symbol='circle-open'),
            legendgroup=team,
            visible=True if team == comparison_df['team'].unique()[0] else 'legendonly'
        ))
    
    fig.update_layout(
        title='Average Points Scored Per Team Over the Years: Actual vs Predicted',
        title_font=dict(size=24, family='Arial', color='white'),
        xaxis=dict(
            title='Season',
            title_font=dict(size=16, color='white'),
            tickfont=dict(size=14, color='white'),
            showgrid=True,
            gridcolor='rgba(255,255,255,0.2)',
            tickangle=45
        ),
        yaxis=dict(
            title='Average Points Per Drive',
            title_font=dict(size=16, color='white'),
            tickfont=dict(size=14, color='white'),
            showgrid=True,
            gridcolor='rgba(255,255,255,0.2)',
        ),
        updatemenus=[
            {
                'buttons': [
                    {
                        'label': team,
                        'method': 'update',
                        'args': [
                            {'visible': [(team == trace.legendgroup) for trace in fig.data]},
                            {'title': f'Average Points for {team}'}
                        ]
                    }
                    for team in comparison_df['team'].unique()
                ],
                'direction': 'down',
                'showactive': True,
                'x': 1.3,
                'xanchor': 'left',
                'y': 1.15,
                'yanchor': 'top',
                'font': dict(color='white')
            }
        ],
        height=700, 
        width=1000,
        plot_bgcolor='rgba(20, 20, 20, 1)',
        paper_bgcolor='rgba(0, 0, 0, 1)',
        legend=dict(
            font=dict(size=14, color='white'),
            bgcolor='rgba(50, 50, 50, 0.8)',
        )
    )
    
    st.plotly_chart(fig)
