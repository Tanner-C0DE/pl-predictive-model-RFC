import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib
import matplotlib.pyplot as plt

# 1. LOAD AND MERGE BOTH SEASONS
def load_data(file):
    d = pd.read_csv(file)
    d.columns = d.columns.str.strip()
    d = d.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
    d['Date'] = pd.to_datetime(d['Date'], dayfirst=True, errors='coerce')
    return d.dropna(subset=['Date', 'HomeTeam', 'AwayTeam'])

df24 = load_data('24-25 EPL data.csv')
df25 = load_data('25-26 EPL data.csv')

# Combine into one master timeline
full_df = pd.concat([df24, df25]).sort_values('Date').reset_index(drop=True)

def get_stats(team_name, until_date, data_df):
    """Looks back at all available history (24/25 + 25/26) up to the match date."""
    history = data_df[((data_df['HomeTeam'] == team_name) | (data_df['AwayTeam'] == team_name)) & (data_df['Date'] < until_date)].copy()
    
    # If it's a promoted team with no history yet, use league baseline
    if len(history) < 2: 
        return {'ppg': 1.0, 'form': 1.0, 'gd': 0.0, 'rest': 7, 'eff': 0.3, 'def': 1.5, 'cards': 2.0}
    
    l10 = history.tail(10)
    pts = sum([3 if m['FTR'] == ('H' if m['HomeTeam'] == team_name else 'A') else (1 if m['FTR'] == 'D' else 0) for _, m in l10.iterrows()])
    gf = sum([m['FTHG'] if m['HomeTeam'] == team_name else m['FTAG'] for _, m in l10.iterrows()])
    ga = sum([m['FTAG'] if m['HomeTeam'] == team_name else m['FTHG'] for _, m in l10.iterrows()])
    
    l5 = history.tail(5)
    f_pts = sum([3 if m['FTR'] == ('H' if m['HomeTeam'] == team_name else 'A') else (1 if m['FTR'] == 'D' else 0) for _, m in l5.iterrows()])
    t_shots = sum([m['HS'] if m['HomeTeam'] == team_name else m['AS'] for _, m in l5.iterrows()])
    t_sot = sum([m['HST'] if m['HomeTeam'] == team_name else m['AST'] for _, m in l5.iterrows()])
    
    return {
        'ppg': pts/len(l10), 
        'form': f_pts/5, 
        'gd': (gf-ga)/len(l10), 
        'rest': min((until_date - history['Date'].max()).days, 10), 
        'eff': t_sot/t_shots if t_shots > 0 else 0.3, 
        'def': ga/len(l10), 
        'cards': sum([(m['HY'] + m['HR']*2) if m['HomeTeam'] == team_name else (m['AY'] + m['AR']*2) for _, m in history.tail(3).iterrows()])
    }

# 2. TRAIN THE RFC ON ALL DATA
features, targets = [], []
print("Training Multi-Season RFC Model...")
for i, row in full_df.iloc[40:].iterrows(): 
    h, a = get_stats(row['HomeTeam'], row['Date'], full_df), get_stats(row['AwayTeam'], row['Date'], full_df)
    features.append([h['ppg'], a['ppg'], h['form'], a['form'], h['gd'], a['gd'], h['rest'], a['rest'], h['eff'], a['eff'], h['def'], a['def'], h['cards'], a['cards']])
    targets.append(row['FTR'])

le = LabelEncoder()
y = le.fit_transform(targets) 
rf = RandomForestClassifier(n_estimators=200, max_depth=7, random_state=42)
rf.fit(features, y)

def predict_match(home, away, date_str):
    target_date = pd.to_datetime(date_str, format='%m/%d/%y')
    h = get_stats(home, target_date, full_df)
    a = get_stats(away, target_date, full_df)
    
    vec = [[h['ppg'], a['ppg'], h['form'], a['form'], h['gd'], a['gd'], h['rest'], a['rest'], h['eff'], a['eff'], h['def'], a['def'], h['cards'], a['cards']]]
    probs = rf.predict_proba(vec)[0]
    
    factors = {
        'Strength (PPG)': (round(h['ppg'],2), round(a['ppg'],2)), # Points per game
        'Form (Last 5 games)': (round(h['form'],2), round(a['form'],2)), # Recent form
        'Goal Diff': (round(h['gd'],2), round(a['gd'],2)), # Average goal difference
        'Rest Days': (h['rest'], a['rest']), # Days since last match
        'Efficiency': (round(h['eff'], 3), round(a['eff'], 3)), # Shot on target efficiency
        'Defense Rating': (round(h['def'], 2), round(a['def'], 2)), # Average goals conceded
        'Cards Penalty': (h['cards'], a['cards']) # Recent disciplinary record
    }
    
    print(f"\n=== MATCH ANALYSIS (Multi-Season RFC): {home} vs {away} ({date_str}) ===")
    print(f"{'FACTOR':<21} | {home:<15} | {away:<15}")
    print("-" * 55)
    for f, val in factors.items():
        print(f"{f:<21} | {val[0]:<15} | {val[1]:<15}")
    print("-" * 55)
    print(f"{'WIN PROBABILITY':<21} | {probs[2]:<15.2%} | {probs[0]:<15.2%}")
    print(f"{'DRAW PROBABILITY':<21} | {probs[1]:<15.2%}")
    
    pred_idx = np.argmax(probs)
    res_map = {2: f"{home} Wins!", 0: f"{away} Wins!", 1: "It's a Draw!"}
    print(f"\n>>> PREDICTION: {res_map[pred_idx]} <<<")


# 1. SAVE THE MODEL 
joblib.dump(rf, 'epl_model_v1.pkl')
joblib.dump(le, 'label_encoder.pkl')
print("Model saved as epl_model_v1.pkl")

# 2. PLOT FEATURE IMPORTANCE
importances = rf.feature_importances_
feat_names = ['H_PPG', 'A_PPG', 'H_Form', 'A_Form', 'H_GD', 'A_GD', 'H_Rest', 'A_Rest', 'H_Eff', 'A_Eff', 'H_Def', 'A_Def', 'H_Cards', 'A_Cards']
indices = np.argsort(importances)

plt.figure(figsize=(10, 6))
plt.title('Which Stats Matter Most to the RFC?')
plt.barh(range(len(indices)), importances[indices], color='skyblue')
plt.yticks(range(len(indices)), [feat_names[i] for i in indices])
plt.xlabel('Relative Influence')
plt.tight_layout()
plt.savefig('feature_importance.png')

# --- CUSTOM PREDICTION INPUT ---
home_team = input("Enter home team: ")
away_team = input("Enter away team: ")
date_input = input("Enter date (MM/DD/YY): ")
predict_match(home_team, away_team, date_input)

# Teams in the 2025-2026 EPL Season:
# Liverpool
# Bournemouth
# Aston Villa
# Arsenal
# Newcastle
# Nott'm Forest
# Man City
# Man United
# Chelsea
# Tottenham
# Everton
# Brentford
# Crystal Palace
# Fulham
# West Ham
# Wolves
# Leeds
# Brighton
# Burnley
# Sunderland

# Teams in the 2024-2025 EPL Season and not in 2025-2026:
# Southampton
# Leicester City
# Ipswich Town