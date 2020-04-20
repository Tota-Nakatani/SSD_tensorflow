#%%
import numpy as np 
import pickle

dataset_path = '/Users/nakatanitota/Desktop/code/ObjectDetection_SSD_TensorFlow/'
with open( dataset_path + 'VOC2007.pkl', 'rb' ) as file:
        data = pickle.load( file )      # [ ファイル名, N * 24 次元の配列 ]
        keys = sorted( data.keys() )    # ファイル名のリスト : []


# %%
b=data['000010.jpg']
b.shape
#%%
def load_image_voc2007( path ):
    """
    load specified image

    Args: image path
    Return: resized image, its size and channel
    """
    

    img = imread( path )
    h, w, c = img.shape
    img = imresize( img, (300, 300) )
    img = img[:, :, ::-1].astype('float32')
    img /= 255.
    return img, w, h, c

n_trains = int( round(0.8 * len(keys)) )


    # トレーニングデータとテストデータの key
train_keys = keys[:n_trains]
a=load_image_voc2007(  dataset_path + train_keys[0] )


# %%
from imageio import imread

# %%
