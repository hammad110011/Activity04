from flask import Flask, request, jsonify
import joblib  # Use joblib or another library to load your model

app = Flask(__name__)

# Load your trained model
model = joblib.load("model/bertbinaryclassifier.py")

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    prediction = model.predict([data['input']])
    return jsonify({'prediction': prediction.tolist()})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
