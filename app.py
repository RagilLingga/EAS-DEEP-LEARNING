from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd

app = Flask(__name__)

# Memuat model
model = joblib.load('knn_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Mendapatkan data dari request
    data = request.get_json(force=True)
    df = pd.DataFrame([data])
    
    # Melakukan prediksi
    prediction = model.predict(df)
    
    # Mengembalikan hasil prediksi
    return jsonify({'prediction': int(prediction[0])})

if __name__ == '__main__':
    app.run(debug=True)
