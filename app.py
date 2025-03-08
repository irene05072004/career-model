from flask import Flask, request, jsonify, render_template
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC

app = Flask(__name__)


df = pd.read_csv('cs_students.csv')
df.columns = df.columns.str.strip()  


model = joblib.load('career_guidance_pipeline.pkl')
encoders = joblib.load('encoders.pkl')
scaler = joblib.load('scaler.pkl')

@app.route('/get_options', methods=['GET'])
def get_options():
    
    options = {column: encoder.classes_.tolist() for column, encoder in encoders.items()}
    return jsonify(options)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    input_data = {
        'Gender': request.form['Gender'],
        'Age': float(request.form['Age']),
        'GPA': float(request.form['GPA']),
        'Major': request.form['Major'],
        'Interested Domain': request.form['Interested Domain'],
        'Projects': request.form['Projects'],
        'Python': request.form['Python'],
        'SQL': request.form['SQL'],
        'Java': request.form['Java']
    }

    
    for column in ['Gender', 'Major', 'Interested Domain', 'Projects', 'Python', 'SQL', 'Java']:
        input_data[column] = encoders[column].transform([input_data[column]])[0]

   
    input_features = pd.DataFrame([input_data])
    input_features[['Age', 'GPA']] = scaler.transform(input_features[['Age', 'GPA']])

    prediction = model.predict(input_features)
    predicted_label = encoders['Future Career'].inverse_transform(prediction)[0]

   
    return render_template('result.html', predicted_label=predicted_label)

    
if __name__ == '__main__':
    app.run(debug=True)
