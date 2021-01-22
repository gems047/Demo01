#import pandas as pd

from werkzeug.utils import secure_filename
from flask import Flask, request , redirect , render_template 

from Inception_Network_Query import image_semantic
from keras import backend as K


from annoy import AnnoyIndex
import random
import numpy as np
import os

# load model

def load_model():
    ann_index = AnnoyIndex(2048, 'hamming')
    ann_index.load('vgdata.ann')
    return ann_index

# app
app = Flask(__name__)
app.config["IMAGE_UPLOADS"] = "/home/user/Monsoon_Courses_2020/Research/Presentation_Results/deploy_heroku/img_upload"
app.config["ALLOWED_IMAGE_EXTENSIONS"] = [ "JPG", "PNG" ]

def allowed_image(filename):

    if not "." in filename:
        return False

    ext = filename.rsplit(".", 1)[1]

    if ext.upper() in app.config["ALLOWED_IMAGE_EXTENSIONS"]:
        return True
    else:
        return False

@app.route('/')
def upload_file():
   return render_template('index.html')
 
@app.route('/uploader', methods = ['GET', 'POST'])
def uploader(): 
    if request.method == 'POST':
    
        f = request.files['file']
        if f.filename == "":
                print("No filename")
                return redirect(request.url)

        if allowed_image(f.filename):
            
            fname = secure_filename(f.filename)
            f.save(os.path.join(app.config["IMAGE_UPLOADS"], fname))
            #f.save(filename)
            img = f.filename
            
            print( 'printing' , img )
            #return redirect(request.url)
            
        else:
            print("That file extension is not allowed")
            return redirect(request.url)
    
            
    return render_template('index.html')

# routes
@app.route('/query', methods=['GET', 'POST'])
def query():
        fname_list = os.listdir(app.config["IMAGE_UPLOADS"])
        #filename = fname_list[0]
        
        for fl in fname_list:
        
            file_path = app.config["IMAGE_UPLOADS"]+'/' + fl
    
            ext = fl.rsplit(".", 1)[1]
            print('test')
            output = {'No query image'}
            if ext.upper() in app.config["ALLOWED_IMAGE_EXTENSIONS"]:
                #conv                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   ert image data into feature
                image_feature = image_semantic(file_path)
                K.clear_session()
                
                print('inside if')
                img_indx = np.load('image_ids_list.npy')
                ann_index = load_model()
                search_results_pos = ann_index.get_nns_by_vector(image_feature[0], n=10, search_k = 1000,  include_distances=False)

                search_results_img  = []
                for j in search_results_pos:
                    img_name = img_indx[j] + '.jpg'
                    search_results_img.append(img_name)
                
                print("i am here")
                # send back to browser
                #output = {'results': int(result[0])}
                output = {'10 results': search_results_img}

                # return data
                os.remove(file_path)
                
                #return render_template('index.html' , results = output )
            else:
                
                print('file extension not allowed')
                #return render_template('index.html' )
                
        return render_template('index.html' , results = output )

if __name__ == '__main__':
    app.run(port = 5000, debug=True)
