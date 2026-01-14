from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

# Load the trained model
model = joblib.load('symptom_model.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        symptoms = data.get('symptoms', '')
        
        if not symptoms:
            return jsonify({'error': 'No symptoms provided'}), 400

        # Predict
        prediction = model.predict([symptoms])[0]
        
        # Get Confidence Probability (optional but nice for thesis)
        probabilities = model.predict_proba([symptoms]).max()

        return jsonify({
            'specialty': prediction,
            'confidence': float(probabilities),
            'source': 'Local ML Model'
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("🚀 ML Service running on port 5001")
    app.run(port=5001)