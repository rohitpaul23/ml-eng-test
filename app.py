from flask import Flask, request, jsonify, send_file
import os
from pdf2image import convert_from_path
import numpy as np
import cv2
from werkzeug.utils import secure_filename
import wall  # Assuming 'wall.py' contains the 'main' function for wall and room detection
import room
import logging

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.DEBUG)

# Create a folder for saving uploaded files
UPLOAD_FOLDER = './uploads'
RESULT_FOLDER = './results'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER

# Allow only certain file extensions
ALLOWED_EXTENSIONS = {'pdf', 'png', 'jpg', 'jpeg'}

def image2pdf(pdf_path):
    
    images = convert_from_path(pdf_path)

    # Process each extracted image
    for page_number, img in enumerate(images):
        # Convert PIL image to OpenCV format
        image = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
        
    return image


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def home():
    return "Wall and Room Annotation API"

@app.route('/run-inference', methods=['POST'])
def runInference():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if file and allowed_file(file.filename):
        # Secure the filename and save it
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        if file.filename.rsplit('.', 1)[1].lower() == 'pdf':
            input_image = image2pdf(file_path)
        else:
            input_image = cv2.imread(file_path)
            input_image = cv2.cvtColor(input_image, cv2.COLOR_RGB2GRAY)
        
        # Call the main function in 'wall.py' for processing
        try:
            # Get the task type from the query parameter, defaulting to 'wall' if not provided
            task_type = request.args.get('type', 'wall')

            # Log the incoming task type
            logging.debug(f"Processing task type: {task_type}")
            
            # Call different functions based on the task type
            if task_type == 'wall':
                # Assuming 'main' in wall.py handles wall detection
                annotated_image = wall.main(input_image)
            elif task_type == 'room':
                # Assuming 'secondary_function' in wall.py handles room detection
                wall_image = wall.main(input_image)
                annotated_image = room.main(wall_image, input_image)
            else:
                return jsonify({'error': f"Unknown type '{task_type}'. Supported types: 'wall', 'room'."}), 400
                
            
            # Define result_path after processing to ensure it is always set
            result_filename = "annotated_" + filename.rsplit('.', 1)[0] + ".png"
            result_path = os.path.join(app.config['RESULT_FOLDER'], result_filename)
            logging.debug(f"Saving annotated image to: {result_path}")
            
            cv2.imwrite(result_path, annotated_image)
            
            
            if not os.path.exists(result_path):
                logging.error(f"Result image not found at: {result_path}")
            
            
            # Return the annotated image as a response
            return send_file(result_path, mimetype='image/png')
        
        except Exception as e:
            logging.error(f"Error during processing: {str(e)}")
            return jsonify({'error': str(e)}), 500
    
    return jsonify({'error': 'Unsupported file type'}), 400

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
