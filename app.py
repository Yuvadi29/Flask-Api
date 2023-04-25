from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

app = Flask(__name__)
CORS(app)

startup_data = pd.read_csv('mpj.csv')
startup_data = startup_data.dropna()

# Calculate Z-score for 'Amount in USD' column
z_scores = (startup_data['Amount in USD'] - startup_data['Amount in USD'].mean()) / startup_data['Amount in USD'].std()

# Define a threshold for Z-score (e.g. Z-score > 3 or < -3)
z_score_threshold = 3

# Filter out rows where Z-score is greater than the threshold
startup_data = startup_data[abs(z_scores) <= z_score_threshold]

le_industry = LabelEncoder()
le_city = LabelEncoder()
le_investor = LabelEncoder()
le_investment_type = LabelEncoder()

startup_data['Industry Vertical'] = le_industry.fit_transform(startup_data['Industry Vertical'])
startup_data['City  Location'] = le_city.fit_transform(startup_data['City  Location'])
startup_data['Investors Name'] = le_investor.fit_transform(startup_data['Investors Name'])
startup_data['InvestmentnType'] = le_investment_type.fit_transform(startup_data['InvestmentnType'])

X = startup_data[['Industry Vertical', 'City  Location', 'InvestmentnType']].values
y = startup_data['Amount in USD'].values

reg_model = LinearRegression()
reg_model.fit(X, y)

r2_score = reg_model.score(X, y)

nn_model = NearestNeighbors(n_neighbors=15, algorithm='ball_tree')
nn_model.fit(X)

@app.route('/')
def home():
    return 'Startup Funding'
    
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    user_industry = data['industry_vertical']
    user_city = data['city']
    user_investment_type = data['investment_type']
    
    if user_industry not in le_industry.classes_:
        return jsonify({'error': 'Invalid input for industry vertical.'})
    if user_city not in le_city.classes_:
        return jsonify({'error': 'Invalid input for city.'})
    if user_investment_type not in le_investment_type.classes_:
        return jsonify({'error': 'Invalid input for investment type.'})
    
    user_industry_encoded = le_industry.transform([user_industry])[0]
    user_city_encoded = le_city.transform([user_city])[0]
    user_investment_type_encoded = le_investment_type.transform([user_investment_type])[0]
    
    new_data_point = [[user_industry_encoded, user_city_encoded, user_investment_type_encoded]]
    _, indices = nn_model.kneighbors(new_data_point)

    output_data = startup_data.iloc[indices[0]]
    
    # Add a new column for predicted profit
    output_data['Predicted Amount'] = reg_model.predict(output_data[['Industry Vertical', 'City  Location', 'InvestmentnType']].values)
    
    # Calculate profit/loss percentage
    output_data['Profit/Loss %'] = (output_data['Predicted Amount'] - output_data['Amount in USD']) / output_data['Amount in USD'] * 100
    
    # Add a new column indicating whether there will be profit or loss
    output_data['Outcome'] = output_data.apply(lambda row: 'Profit' if row['Predicted Amount'] > row['Amount in USD'] else 'Loss', axis=1)
    
    return output_data.to_json(orient='records')

if __name__ == "__main__":
    app.run(debug=True)