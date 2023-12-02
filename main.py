from tensorflow import keras
import tensorflow as tf
import numpy as np
import cv2
from flask_cors import CORS
from flask import Flask, request, jsonify

# Load the model
model = keras.models.load_model("endoscopy-classification-freezing-90%.h5")


def transformImage(img_array):
    # Resize the image
    img_resize_rgb = cv2.resize(img_array, (224, 224))

    # Add batch dimension
    data = np.expand_dims(img_resize_rgb, axis=0)

    return data


def predict(x):
    class_names = [
        "dyed-lifted-polyps",
        "dyed-resection-margins",
        "esophagitis",
        "no-category",
        "normal-cecum",
        "normal-pylorus",
        "normal-z-line",
        "polyps",
        "ulcerative-colitis"
    ]
    # Make predictions
    predictions = model.predict(x)

    # Get the predicted label index
    pred_index = np.argmax(predictions[0])

    # Get the actual class name
    predicted_class = class_names[pred_index]

    # Get the confidence score for the predicted class
    confidence_score = predictions[0][pred_index]

    return predicted_class, confidence_score


app = Flask(__name__)
CORS(app)

@app.route("/", methods=["GET", "POST"])

def index():
    if request.method == "POST":
        file = request.files.get('file')
        if file is None or file.filename == "":
            return jsonify({"error": "no file "})
        try:
            img_array = cv2.imdecode(np.frombuffer(file.read(), np.uint8), -1)
            tensor = transformImage(img_array)
            prediction, confidence = predict(tensor)
            
            # Multiply confidence by 100 to express it as a percentage
            confidence_percentage = float(confidence) * 100
            
            data = {"prediction": prediction, "confidence": confidence_percentage}
            return jsonify(data)
        except Exception as e:
            return jsonify({"error": "no file "})

    return "OK"


if __name__ == "__main__":
    app.run(debug=True)
