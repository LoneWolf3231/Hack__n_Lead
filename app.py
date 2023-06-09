from flask import Flask, request, jsonify, render_template
import tensorflow as tf

app = Flask(__name__)

# Load the models when the application starts
model1 = tf.keras.models.load_model('C:\Users\shashidhar\Downloads\combined_notebook.ipynb')
model2 = tf.keras.models.load_model('C:\Users\shashidhar\Downloads\Fake_News_Detector.ipynb')

# Preprocessing function for input data
def preprocess_data(data):
    preprocessed_data=data

    return preprocessed_data

# Define API endpoints
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict_model1', methods=['POST'])
def predict_model1():
    data = request.json['data']
    preprocessed_data = preprocess_data(data)

    # Perform inference using model1
    prediction = model1.predict(preprocessed_data)

    # Format the response
    response = {'prediction': prediction.tolist()}

    return jsonify(response)

@app.route('/predict_model2', methods=['POST'])
def predict_model2():
    data = request.json['data']
    preprocessed_data = preprocess_data(data)

    # Perform inference using model2
    prediction = model2.predict(preprocessed_data)

    # Format the response
    response = {'prediction': prediction.tolist()}

    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)