from werkzeug.utils import secure_filename
from flask import Flask, request , redirect , render_template , url_for

from Inception_Network_Inmemory import image_semantic
from keras import backend as K

from annoy import AnnoyIndex
import random
import numpy as np
import cv2
import sys

# load model

def load_model():
    ann_index = AnnoyIndex(2048, 'hamming')
    ann_index.load('vgdata.ann')
    return ann_index

# app
app = Flask(__name__)

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
def index():
    return render_template('index.html')
 
# routes
@app.route('/query', methods=['GET', 'POST'])
def query():
         
    output = {'No query image'}      
    if request.method == 'POST':
    
        f = request.files['file']
        
        
        if f.filename == "":
            print("No filename")
            return redirect(request.url)

        if allowed_image(f.filename):
            
            print(request.files , file=sys.stderr)
            print('image file' , type(f.filename) , type(f))
            
            #byte file
            fl = f.read()
           
            npimg = np.fromstring(fl, np.uint8)
            print(type(npimg) , npimg)
            img = cv2.imdecode(npimg,cv2.IMREAD_COLOR)
            print(img.shape , type(img))
            img_resize = cv2.resize(img, (299,299))
            
            image_feature = image_semantic(img_resize)
            K.clear_session()
           
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
            
        else:
            print("That file extension is not allowed")
            return redirect(request.url)
            
    return render_template('index.html' , results = output )
    
@app.route('/home')
def home():
    # do something
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(port = 5000, debug=True)    
