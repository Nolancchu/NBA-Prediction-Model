import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import joblib

# Load your trained model and scaler
model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')
selected_features = ['OFFRTG_diff','DEFRTG_diff','NETRTG_diff','AST%_diff','EFG%_diff','TS%_diff','USG%_diff','PACE_diff','PTS_diff','REB_diff','AST_diff','STL_diff','BLK_diff','TOV_diff','+/-_diff']

def predict_win_probability(model, scaler, home_stats, away_stats):
    matchup = pd.DataFrame()
    for stat in selected_features:
        base_stat = stat.replace('_diff', '')
        matchup[stat] = [home_stats[base_stat] - away_stats[base_stat]]
    
    matchup_scaled = scaler.transform(matchup)
    probability = model.predict_proba(matchup_scaled)[0][1]  # Probability of home team winning
    return probability

def dictate_line(model, scaler, home_stats, away_stats):
    prob = predict_win_probability(model, scaler, home_stats, away_stats)
    
    if prob >= .5:
        return -(prob/(1-prob))*100
    else:
        return ((1-prob)/prob) *100