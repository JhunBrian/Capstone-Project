from datetime import datetime
# from geopy.geocoders import Nominatim
import os
import numpy as np # 1.23.5
import matplotlib.pyplot as plt # 3.6.3
import cv2 # 4.7.0.68
import csv

from tensorflow import keras # 2.11.0
import tensorflow as tf # 2.11.0
import tensorflow_hub as hub # 0.12.0
from keras.models import load_model # 2.11.0
from warnings import filterwarnings as fws
fws('ignore')

leaf_segmentor = load_model('models/LEAF_SEGMENTOR/normalized_adam-v1.h5', compile=False)
disease_segmentor = load_model('models/DISEASE_SEGMENTOR/81_resnet50-disease-segmentor.h5', compile=False)

classifier = load_model('models/CLASSIFIER/DenseNet169-v3.h5', custom_objects={'KerasLayer':hub.KerasLayer})
classifier.compile(optimizer='adam', 
                   loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), 
                   metrics=['acc'])

def normalizer(image_array):
    normed = cv2.normalize(image_array, None, 0, 1.0, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    return normed

def predict_mask(image_array, threshold=0.6):
    image = cv2.resize(image_array, (256, 256))
    expand = np.expand_dims(normalizer(image), axis=0)
    prediction = leaf_segmentor.predict(expand, verbose='silent')
    pred_image = np.squeeze(prediction > threshold)
    return pred_image.astype('uint8')

def remove_bg(image_array, mask):
    image = cv2.resize(image_array, (256, 256))
    mask = np.where((mask==0), 0, 1).astype('uint8')
    result = image * mask[:, :, np.newaxis]
    return result

def classify(image_array):
    class_dict = {0:'Bacterial Blight', 1:'Brown Spot', 2:'Rice Blast', 3:'Tungro Virus'}
    normalize = normalizer(image_array)
    expand_array = np.expand_dims(normalize, axis=0)
    predict = classifier.predict(expand_array, verbose='silent')
    confidence = predict.max() * 100
    pred_class = class_dict[np.round(predict, 3).argmax()]
    return np.round(confidence, 2), pred_class

def predict_severity(image_array, threshold=0.3):
    expand = np.expand_dims(normalizer(image_array), axis=0)
    prediction = disease_segmentor.predict(expand, verbose='silent')
    pred_image = np.squeeze(prediction > threshold)
    return pred_image.astype('uint')

def ses_evaluation(proportion):
    if proportion < 1:
        class_ = 0
    elif proportion > 1 and proportion < 4:
        class_ = 1
    elif proportion > 3 and proportion < 11:
        class_ = 2
    elif proportion > 10 and proportion < 26:
        class_ = 3
    elif proportion > 25 and proportion < 51:
        class_ = 4
    elif proportion > 50 and proportion < 76:
        class_ = 5
    else:
        class_ = 6
        
    return class_

def calculate_severity(leaf_mask, disease_mask):
    leaf_area = np.sum(leaf_mask == 1)
    disease_area = np.sum(disease_mask == 1)
    proportion = (disease_area / leaf_area) * 100
    return np.round(proportion, 2)

def append_to_csv(file_name, data):
    fieldnames = data.keys()
    if not os.path.isfile(file_name):
        with open(file_name, mode='w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerow(data)
    else:
        with open(file_name, mode='a', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writerow(data)
            
def overlay_images(img1, img2, name, alpha=0.4):
    img1 = cv2.cvtColor(img1.astype('uint8'), cv2.COLOR_BGR2RGB)
    img2 = img2.astype('uint8')
    img2 = cv2.merge((img2, img2, img2)) * 255
    beta = alpha
    overlay = cv2.addWeighted(img1, alpha, img2, beta, 0)
    cv2.imwrite(f'static/{name}_ann.jpg', cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    return overlay

def sequential_process(image_array, name=None):
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if os.path.exists(image_array):
        image_array = cv2.imread(image_array)
    elif type(image_array) == np.ndarray:
        pass
    
    original_image = cv2.resize(image_array, (256, 256))
    
    step2 = predict_mask(image_array)
    step3 = remove_bg(original_image, step2)
    step4 = classify(step3)
    step5 = predict_severity(step3)
    
    conf, label = step4
    severity = calculate_severity(step2, step5)
    overlay_images(step3, step5, name=name)
    
    log_dict = {'DateTime':current_time,
                'label':"Healthy" if ses_evaluation(severity) == 0 else label, 
                'confidence':conf, 
                'SES class':"Not Supported" if label == "Tungro Virus" else ses_evaluation(severity), 
                'severity':severity}
    
    append_to_csv('logs.csv', log_dict)
    ann_path = f'static/{name}_ann.jpg'
    return log_dict, ann_path