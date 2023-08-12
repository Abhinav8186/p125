from flask import Flask,jsonify,request
from Classifier import get_prediction

@app.route("/predict-alphabet", methods=["POST"])

def predict_data():
    image = request.files.get("alphabet")
    prediction = get_prediction(image)
    return jsonify({
        "prediction":prediction
    }),200