from flask import Flask, request, render_template, send_from_directory
import os
from werkzeug.utils import secure_filename
from model import predict_image  # Import the ML model function

app = Flask(__name__)

# Folder to store uploaded images
UPLOAD_FOLDER = 'static/uploaded_images'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Allowed extensions for image upload
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

# Function to check allowed file extensions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Route to handle image uploads
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'image' not in request.files:
            return "No file part", 400
        file = request.files['image']
        if file.filename == '':
            return "No file selected", 400
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # Predict the image using the ML model
            prediction = predict_image(filepath)

            # Return the result with the image and prediction
            return render_template('index.html', filename=filename, prediction=prediction)
    return render_template('index.html')

# Route to serve uploaded images
@app.route('/uploaded_images/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    app.run(debug=True)
