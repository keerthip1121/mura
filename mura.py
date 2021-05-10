# -*- coding: utf-8 -*-
"""
Created on Thu May  6 11:07:09 2021

@author: keerthi
"""


# -*- coding: utf-8 -*-
"""
Created on Mon Mar  1 11:17:58 2021

@author: keerthi
"""


from flask import Flask, render_template, request
import numpy as np
from keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array

app = Flask(__name__, template_folder="template")

label = {0:'negative',1:'positive'}

model = load_model('./ensemble_mura.h5')
 
@app.route("/", methods=["GET"])
def home():
    return render_template("page.html")


def predict(path):
    img = load_img(path, color_mode='rgb',target_size=(224,224))
    img = img_to_array(img)
    img = np.array([img])
    prediction = model.predict(img)
    if prediction<0.5:
        text = label[0]
    else:
        text = label[1]

    return text

predict(r'D:/internship/MURA/static/image2.png')

@app.route("/submit", methods=["POST"])

def deploy():
    image = request.files["my_img"]
    path = "static/" + image.filename
    image.save(path)
    
    pred = predict(path)
    
    return render_template("page.html", prediction = pred, img_path = path)

if __name__=='__main__':
    app.run(debug=True)

    
    
    
