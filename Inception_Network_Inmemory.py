#from keras.applications import InceptionV3
#https://stackoverflow.com/questions/48232331/norm-parameters-in-sklearn-preprocessing-normalize


from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.applications.inception_v3 import preprocess_input

from sklearn import preprocessing
 
import numpy as np


def image_semantic(img_data):
    #img_data is coming from web request and not image file on disk
    print(type(img_data))
    model = InceptionV3(weights='imagenet', include_top=False, pooling = 'avg')
    #img = image.load_img(img_path, target_size=(299, 299))
    #img_data = image.img_to_array(img)
    img_data = np.expand_dims(img_data, axis=0)
    img_data = preprocess_input(img_data)

    incept_feature = model.predict(img_data)

    incept_l2norm = preprocessing.normalize(incept_feature.reshape(1,-1) , norm = 'l2')
    
    return incept_l2norm
    
#img_path = 'im.jpg'
#incept_l2norm = image_semantic(img_path)


