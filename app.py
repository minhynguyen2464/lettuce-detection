import onnxruntime as ort
import numpy as np
from PIL import Image
from torchvision import transforms
from flask import Flask, request, jsonify
import io

app = Flask(__name__)

# Load the ONNX model
session = ort.InferenceSession('./working/lettuce_npk.onnx')

# Define class names
classes = ['Thiếu hụt Kali -K', 'Thiếu hụt Nito N-', 'Thiếu hụt photpho -P', 'Đầy đủ dinh dưỡng']
class_names = dict(zip(range(len(classes)), sorted(classes)))

def preprocess_image(image_data):
    """
    The `preprocess_image` function takes image data, resizes it, converts it to a tensor, and returns
    it as a numpy array.
    
    :param image_data: The `image_data` parameter is the raw image data that you want to preprocess.
    This data could be in the form of bytes representing an image file. The `preprocess_image` function
    takes this raw image data, applies a series of transformations to it using the `transforms.Compose`
    method from the
    :return: The `preprocess_image` function returns a NumPy array representing the preprocessed image
    data.
    """
    t = transforms.Compose([
        transforms.Resize((224, 224), antialias=True),
        transforms.ToTensor(),
    ])
    image = Image.open(io.BytesIO(image_data)).convert('RGB')
    image = t(image).unsqueeze(0)  # Add batch dimension
    return image.numpy()

def get_prediction(image_data):
    """
    The function `get_prediction` takes image data, preprocesses it, runs it through a model session,
    and returns the predicted class label based on the highest score.
    
    :param image_data: It looks like the `get_prediction` function takes image data as input,
    preprocesses the image, runs it through a neural network model using a session, and returns the
    predicted class label based on the highest score from the model's output
    :return: The function `get_prediction` returns the predicted class label for the input image data
    after processing and running it through a neural network model.
    """
    input_image = preprocess_image(image_data)
    pred = session.run(None, {'input': input_image})  # Replace 'input' if necessary
    class_label = np.argmax(pred[0])  # Get the index of the class with the highest score
    return class_names[class_label]

# Flask route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    """
    The `predict` function checks for an image in the request, reads the image data, and returns a
    prediction based on the image data.
    :return: The `predict()` function returns a JSON response. If no image is provided in the request,
    it returns an error message with status code 400. If an image is provided, it reads the image data,
    gets a prediction using the `get_prediction()` function, and returns the prediction in a JSON
    response. If an exception occurs during the prediction process, it returns an error message with
    status code
    """
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    image_file = request.files['image']
    image_data = image_file.read()
    
    # Get prediction
    try:
        prediction = get_prediction(image_data)
        return jsonify({'prediction': prediction})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
