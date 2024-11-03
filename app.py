from flask import Flask, request, render_template
from model import predict_win_probability, dictate_line, model, scaler
import pandas as pd
app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])

def home():
    team_stats = (pd.read_csv('updated_stats.csv')).set_index('Team')
    if request.method == 'POST':
        # Get input from the form
        home_team_stats = team_stats.loc[request.form['home_Team']]
        away_team_stats = team_stats.loc[request.form['away_Team']]

        # Make prediction
        line = dictate_line(model, scaler, home_team_stats, away_team_stats)
        percentage = predict_win_probability(model, scaler, home_team_stats, away_team_stats)
        return render_template('result.html', prediction = percentage, probability=line)
    
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)