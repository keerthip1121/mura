# -*- coding: utf-8 -*-
"""
Created on Sun May 16 11:30:50 2021

@author: keerthi
"""


from flask import Flask, render_template, request
import numpy as np
import tensorflow as tf
import cv2

app = Flask(__name__, template_folder="template")

label = {0:'Normal', 1:'Abnormal'} 


model = tf.keras.models.load_model('./basic_cnn.h5')
 
@app.route("/", methods=["GET"])
def home():
    return render_template("page.html")


def predict(path):
    image = cv2.imread(path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2,tileGridSize=(8,8))
    adap_equalized_gray = clahe.apply(gray)

      # Merge channels to create color image (3 channels)
    adap_equalized = cv2.merge([adap_equalized_gray,adap_equalized_gray,adap_equalized_gray])
    img = cv2.resize(adap_equalized,(100,100))
    img = img/255.
    img = np.array([img])
    prediction = model.predict(img)
  
    if prediction<0.5:
        text = 'X-Ray shows '+str(label[0])
    else:
        text = 'X-Ray shows '+str(label[1])
    
    return text



@app.route("/submit", methods=["POST"])

def deploy():
    image = request.files["my_img"]
    path = "static/" + image.filename
    image.save(path)
    
    pred = predict(path)
    
    return render_template("page.html", prediction = pred, img_path = path)

if __name__=='__main__':
    app.run(debug=True)
