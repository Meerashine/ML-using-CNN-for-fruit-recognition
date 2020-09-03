# -*- coding: utf-8 -*-
"""
Created on Thu Aug  6 10:41:15 2020

@author: Meerashine Joe
"""


from flask import Flask, render_template, request, redirect, url_for
#from data import Articles
import os
import matplotlib.image as mpimg
import numpy as np
from PIL import *
from keras.models import load_model
import os.path
import matplotlib.pyplot as plt


app = Flask(__name__)

APP_ROOT = os.path.dirname(os.path.abspath(__file__))

@app.route('/')
def index():
    return render_template('home.html')

@app.route('/about',methods=['GET', 'POST'])
def about():
    return render_template('about.html')


@app.route('/articles')
def articles():
    return render_template('articles.html')

@app.route('/article/<string:id>')
def article(id):
    return render_template('article.html', id =id)

@app.route('/upload', methods =['POST'])
def upload():
    fold = os.path.join(APP_ROOT, '/test_images/')
    print(fold)
    
    if not os.path.isdir(fold):
        os.mkdir(fold)
        
    for file in request.files.getlist("file"):
        print(file)
        filename = file.filename
        destn = "/".join([fold, filename])
        print(destn)
        file.save(destn)
    
    new_destn = os.path.join('test_images'+filename)
    
    train_categories = []
    train_samples = []
    for i in os.listdir(r"C:\Users\Meerashine Joe\Downloads\Nitin george proj\fruits-360\Training"):
       train_categories.append(i)
    print("good to go")
    
    test = Image.open(r"C:\Users\Meerashine Joe\Downloads\Nitin george proj\fruits-360\Test\Apple Braeburn\34_100.jpg")
#test_array = np.array(test)
#plt.imshow(test_array)

    if test.size[0] > test.size[1]:
       scaling = 100 / test.size[1]
       scale_length = int(scaling * test.size[0])
       scale_width = int(scaling * test.size[1])
       scale_image =(scale_length,scale_width)
    
    else:
       scaling = 100 / test.size[0]
       scale_length = int(scaling * test.size[0])
       scale_width = int(scaling * test.size[1])
       scale_image =(scale_length,scale_width)
    
    scale_image_new = test.resize(scale_image)
    scale_image_new_array = np.array(scale_image_new)
    print("image can be viewed")
    plt.imshow(scale_image_new_array)

    a =0
    b =0
    c =100
    d =100
    model = load_model("my model.h5")

    cropped_image = scale_image_new.crop((a,b,c,d))
    cropped_image_array = np.array(cropped_image)
    cropped_image_new =cropped_image_array/255

    test_image = np.reshape(cropped_image_new,newshape=(1,cropped_image_new.shape[0],cropped_image_new.shape[1],cropped_image_new.shape[2]))

    predicted = model.predict(x =test_image)
    prediction = np.argmax(predicted)

    print("The fruit is : ", train_categories[prediction])
    index_1 = np.argsort(predicted)
    predict_final = index_1[:,-3:]
    result = train_categories[predict_final[0][-1]]
    

    return render_template('about.html', results =result)



if __name__ == '__main__':
    #app.run(debug=True)
    app.run(port=5000, debug=True, use_reloader=False,threaded=False)