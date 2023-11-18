from flask import Flask, request, jsonify
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import logging
from logging.handlers import RotatingFileHandler

# Initialize Flask app
app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)
handler = RotatingFileHandler('server.log', maxBytes=10000, backupCount=3)
logger = logging.getLogger('tdm')
logger.setLevel(logging.INFO)
logger.addHandler(handler)

# Load your model and processor
processor = TrOCRProcessor.from_pretrained('model/processor')
model = VisionEncoderDecoderModel.from_pretrained('model/model')

def predict_text(image):
    try:
        print("g")
        pixel_values = processor(images=image, return_tensors="pt").pixel_values
        generated_ids = model.generate(pixel_values)
        return processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    except Exception as e:
        logger.error(f"Error in predict_text: {str(e)}")
        print(f"Error in predict_text: {str(e)}")
        return None

test_image_path = 'test8.jpg'
test_image = Image.open(test_image_path).convert("RGB")
print(predict_text(test_image))

# Define a route for prediction
@app.route('/upload', methods=['POST'])
def predict():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No image provided'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No image selected'}), 400

        image = Image.open(file.stream).convert("RGB")
        text = predict_text(image)

        if text is None:
            raise ValueError("Error in text prediction")

        return jsonify({'extracted_text': text})
    except Exception as e:
        logger.error(f"Error in predict route: {str(e)}")
        return jsonify({'extracted_text': 'An error occurred during prediction'}), 500

# Run the app
if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=8080)


