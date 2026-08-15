
from flask import Flask, request, jsonify

import pickle



app = Flask(__name__)



MODEL_PATH = "model.pkl"



with open(MODEL_PATH, "rb") as file:

    model = pickle.load(file)





@app.route("/", methods=["GET"])

def home():

    return jsonify({

        "message": "Iris Classification API",

        "endpoint": "/predict"

    })





@app.route("/predict", methods=["POST"])

def predict():

    data = request.get_json()



    features = [[

        data["sepal_length"],

        data["sepal_width"],

        data["petal_length"],

        data["petal_width"]

    ]]



    prediction = model.predict(features)[0]



    return jsonify({

        "prediction": prediction

    })





if __name__ == "__main__":

    app.run(host="0.0.0.0", port=5000)

