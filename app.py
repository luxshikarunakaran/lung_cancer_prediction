from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained model
model = joblib.load('lung_cancer_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.form.to_dict()
    features = np.array([list(data.values())]).astype(float)
    prediction = model.predict(features)
    return render_template('result.html', prediction=prediction[0])

if __name__ == '__main__':
    app.run(debug=True)

import speech_recognition as sr


@app.route('/voice_predict', methods=['POST'])
def voice_predict():
    recognizer = sr.Recognizer()
    audio_file = request.files['file']
    audio = sr.AudioFile(audio_file)

    with audio as source:
        audio_data = recognizer.record(source)
        text = recognizer.recognize_google(audio_data)

    # Assume a function to process voice input into features
    features = process_voice_input(text)
    prediction = model.predict(features)
    return jsonify({'prediction': prediction.tolist()})


def process_voice_input(text):
    # Dummy implementation, needs to be replaced with actual processing
    # This function should convert the text to numerical features
    return np.array([[0]])  # Placeholder implementation

