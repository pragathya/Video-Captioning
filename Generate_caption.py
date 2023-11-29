#!/usr/bin/env python
# coding: utf-8

# In[29]:


import numpy as np
import tensorflow as tf

#tf.compat.v1.disable_eager_execution()

#from tensorflow.python.keras.layers import 

from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model,Model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Input, Dense, Dropout, Embedding, LSTM,add


# In[30]:


model_1 = ResNet50(weights='imagenet',input_shape=(224,224,3))
model_1.summary()


# In[31]:


model_1_new = Model(model_1.input,model_1.layers[-2].output)


# In[32]:


def preprocess_img(img):
    img = image.load_img(img,target_size=(224,224))
    img = image.img_to_array(img)
    img = np.expand_dims(img,axis=0)     #Expanding dimensions of image
    
    img = preprocess_input(img)
    
    return img
    


# In[37]:


def encd_img(img):
    img = preprocess_img(img)
    feature_vector = model_1_new.predict(img)
    feature_vector = feature_vector.reshape(1,feature_vector.shape[1])
    return feature_vector


# In[38]:


d = encd_img('Image1784.jpg')
d.shape


# # Importing two dictionaries

# In[39]:


import pickle
with open('models/word_2_idx','rb') as fw:
    word_2_idx = pickle.load(fw)
    
with open('models/idx_2_word','rb') as fi:
    idx_2_word = pickle.load(fi) 


# In[40]:


word_2_idx['the']


# In[41]:


m = load_model('models/my_model.h5')
m.summary()


# In[42]:


def predict_caption(photo):
    
    max_len = 38
    in_text = "<s>"
    for i in range(max_len):
        sequence = [word_2_idx[w] for w in in_text.split() if w in word_2_idx]
        sequence = pad_sequences([sequence],maxlen=max_len,padding='post')
        
        ypred = m.predict([photo,sequence])
        ypred = ypred.argmax() #WOrd with max prob always - Greedy Sampling
        word = idx_2_word[ypred]
        in_text += (' ' + word)
        
        if word == "<e>":
            break
    
    final_caption = in_text.split()[1:-1]
    final_caption = ' '.join(final_caption)
    return final_caption


# In[44]:


predict_caption(d)


# In[45]:


def caption_image(img):
    photo = encd_img(img)
    caption = predict_caption(photo)
    return caption


# In[ ]:




