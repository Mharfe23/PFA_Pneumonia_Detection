from flask import Flask,render_template,request,jsonify
from werkzeug.utils import secure_filename
import os
from tensorflow.keras.models import load_model # type: ignore
import cv2
import numpy as np

Docteurs=[{'nom':'Mohammed Najem', 'ville':'Rabat', 'numero':'0537420201'},
          {'nom':'Taha Filalli', 'ville':'Casablanca', 'numero':'0686764210'},
          {'nom':'Ahmad Ismail', 'ville':'Tanger', 'numero':'0632074114'}]
model = load_model('pretrained_vgg16_model_final.h5')

def preprocess_image(image_path):
    image = cv2.imread(image_path)
    
    # Check if the image was read successfully
    if image is None:
        raise ValueError("Unable to read image at path:", image_path)
    
    # Check if the image has a valid size
    if image.size == 0:
        raise ValueError("Empty image detected at path:", image_path)
    image = cv2.resize(image, (150, 150)) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

def predict_image(image_path):
    processed_image = preprocess_image(image_path)
    predictions = model.predict(processed_image)
    predicted_class = np.argmax(predictions)
    return "Absence" if predicted_class == 0 else "Presence"



app = Flask(__name__,  static_url_path='/static')
app.config['UPLOAD_FOLDER'] = 'upload'
app.config['TEMP_FOLDER'] = 'upload/temp'


@app.route('/')
@app.route('/index.html')
def index():
    return render_template('index.html')



@app.route('/service.html')
def service():
    return render_template('service.html')




@app.route('/predict', methods =['POST'])
def predict():

    if 'xray_file' not in request.files:
        return 'No file part'
    
    file = request.files['xray_file']
    if file.filename == '':
        return 'No selected file'
    
    if file:
        filename = file.filename
        path_file=os.path.normpath(os.path.join(app.config['TEMP_FOLDER'], filename))
        file.save(path_file)
        
    
       
    
    predicted_class = predict_image(path_file)  
    
    subdirectory='normal' if predicted_class=='Absence' else 'pneumonia'
    subdirectory_path = os.path.join(app.config['UPLOAD_FOLDER'],subdirectory)

    if not os.path.exists(subdirectory_path):
        os.makedirs(subdirectory_path)
    
    new_file_path=os.path.join(subdirectory_path,filename)
    os.rename(path_file,new_file_path)
    
    
    

    return render_template('resultat.html',resultat=predicted_class,Docteurs=Docteurs)

if __name__ == '__main__':
    app.run(debug=True)