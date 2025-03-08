import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
import joblib
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

df = pd.read_csv('cs_students.csv')
df.columns = df.columns.str.strip()
df = df.drop(['Student ID', 'Name'], axis=1)

encoders = {}
categorical_columns = ['Gender', 'Major', 'Interested Domain', 'Projects', 'Future Career', 'Python', 'SQL', 'Java']
for column in categorical_columns:
    encoder = LabelEncoder()
    df[column] = encoder.fit_transform(df[column])
    encoders[column] = encoder

joblib.dump(encoders, 'encoders.pkl')

X = df.drop('Future Career', axis=1)
y = df['Future Career']

scaler = StandardScaler()
X[['Age', 'GPA']] = scaler.fit_transform(X[['Age', 'GPA']])

joblib.dump(scaler, 'scaler.pkl')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

pipeline = Pipeline(steps=[
    ('scaler', StandardScaler()),
    ('model', RandomForestClassifier(n_estimators=100, random_state=42))
])
pipeline.fit(X_train, y_train)

joblib.dump(pipeline, 'career_guidance_pipeline.pkl')

@app.route('/get_options', methods=['GET'])
def get_options():
    encoders = joblib.load('encoders.pkl')
    options = {column: encoder.classes_.tolist() for column, encoder in encoders.items()}
    return jsonify(options)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    encoders = joblib.load('encoders.pkl')
    scaler = joblib.load('scaler.pkl')
    pipeline = joblib.load('career_guidance_pipeline.pkl')

    input_data = pd.DataFrame({
        'Gender': [encoders['Gender'].transform([data['Gender']])[0]],
        'Age': [data['Age']],
        'GPA': [data['GPA']],
        'Major': [encoders['Major'].transform([data['Major']])[0]],
        'Interested Domain': [encoders['Interested Domain'].transform([data['Interested Domain']])[0]],
        'Projects': [encoders['Projects'].transform([data['Projects']])[0]],
        'Python': [encoders['Python'].transform([data['Python']])[0]],
        'SQL': [encoders['SQL'].transform([data['SQL']])[0]],
        'Java': [encoders['Java'].transform([data['Java']])[0]]
    })

    predicted_class = pipeline.predict(input_data)
    predicted_label = encoders['Future Career'].inverse_transform(predicted_class)

    return jsonify({'predicted_career': predicted_label[0]})

if __name__ == '__main__':
    app.run(debug=True)


