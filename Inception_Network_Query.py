#from keras.applications import InceptionV3
#https://stackoverflow.com/questions/48232331/norm-parameters-in-sklearn-preprocessing-normalize


from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.applications.inception_v3 import preprocess_input

from sklearn import preprocessing
 
import numpy as np



def image_semantic(img_path):
    #img_path = '000000011699.jpg'
    model = InceptionV3(weights='imagenet', include_top=False, pooling = 'avg')
    img = image.load_img(img_path, target_size=(299, 299))
    img_data = image.img_to_array(img)
    img_data = np.expand_dims(img_data, axis=0)
    img_data = preprocess_input(img_data)

    incept_feature = model.predict(img_data)

    #This gives feature of shape -> (1, 8, 8, 2048)


    #incept_feature1 =np.squeeze(incept_feature)
    #incept_feature2 =np.reshape(incept_feature1, (64,2048))
    #incept_feature3 =np.reshape(incept_feature2, (131072,))

    incept_l2norm = preprocessing.normalize(incept_feature.reshape(1,-1) , norm = 'l2')
    
    return incept_l2norm
    
#img_path = '000000011699.jpg'
#incept_l2norm = image_semantic(img_path)  
#print(incept_l2norm.shape)
#print(incept_l2norm)
 
# This gives feature of shape -> 131072


