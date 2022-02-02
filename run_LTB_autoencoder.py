#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)


# In[2]:


config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)


# In[3]:


#import tensorflow
import pandas as pd


# In[4]:

import sys
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D
import matplotlib.pyplot as plt
from keras import backend as K
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold

# In[5]:


from keras.layers import Dense, Conv2D, Dropout, BatchNormalization, Input, Reshape, Flatten, Conv2DTranspose, MaxPooling2D, UpSampling2D
from keras.models import Model, Sequential
from keras.callbacks import ModelCheckpoint, EarlyStopping


# In[6]:


from PIL import Image, ImageOps
from numpy import array
import numpy as np
import os
import cv2


# In[7]:


from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error


# In[8]:

#SET PARAMETERS
reduced_width,reduced_height=256,256
mtype=sys.argv[1]
if mtype == 'cnn':
    serpentina=False
elif mtype == 'stacked':
    serpentina=True
ae=sys.argv[2]#'normal' #normal, TB, both
metric_eval=sys.argv[3]#'mse'
nepochs = 300
batchsize = 32

# ### Import images

# In[9]:


input_dir='../ChinaSet_AllFiles/CXR_png'
#input_dir=os.path.join(china_path, 'CXR_png')
#input_dir='/content/drive/My Drive/BRICS - TB Latente/Dados/ChinaSet_AllFiles/CXR_png'
#input_dir='/jupyter_dir/Workspace/TB/ChinaSet_AllFiles/CXR_png'
files = sorted([x for x in os.listdir(input_dir) if '.png' in x])


# In[10]:


info_dir='../ChinaSet_AllFiles/ClinicalReadings'
info_files = sorted([x for x in os.listdir(info_dir) if '.txt' in x])


# In[11]:


info_dict={}
info_list_all=[]
for info_file in info_files:
    new_info_list = []
    info_list = open(os.path.join(info_dir,info_file)).readlines()
    info_list = [x.replace('\n','').replace('\t','').replace(' , ',' ').replace(',','').replace('16month','1yrs').replace('64days','0yrs') for x in info_list]

    if len(info_list) ==2:
        #print(info_file, len(info_list),info_list)
        new_info_list.append(info_file.split('.')[0])
        new_info_list.append(info_list[0].split(' ')[0].lower())
        new_info_list.append(info_list[0].split(' ')[1].replace('yrs','').replace('yr',''))
        new_info_list.append(info_list[1].lower())

    elif len(info_list) ==3 and info_list[1] == '':
        #print(info_file, len(info_list),info_list)
        new_info_list.append(info_file.split('.')[0])
        new_info_list.append(info_list[0].split(' ')[0].lower())
        new_info_list.append(info_list[0].split(' ')[1].replace('yrs','').replace('yr',''))
        new_info_list.append(info_list[2].lower())
    else:
        #print(info_file, len(info_list),info_list)
        new_info_list.append(info_file.split('.')[0])
        new_info_list.append(info_list[0].split(' ')[0].lower())
        new_info_list.append(info_list[0].split(' ')[1].replace('yrs','').replace('yr',''))
        new_info_list.append(', '.join(info_list[1:]).lower())

    new_info_list = [x.strip() for x in new_info_list]
    info_dict[info_file.split('.')[0]] = new_info_list
    info_list_all.append(new_info_list)


# In[12]:


nameList=[]
count=0
for file in files:
    #print(count,file)
    count+=1
    nameList.append(file)


# In[13]:


print(len(files))


# ### Transform images
#
#  - Grayscale
#  - Resize to 256x256
#  - Create array of images: nx256x256, where n is the number of images

# In[14]:


def transform_to_1D(m_array):
    m_array[1::2] = m_array[1::2, ::-1]

    return m_array.ravel()


# In[15]:


def back_to_img(arr, m_pixels):
    import copy
    vec = copy.deepcopy(arr)
    size=(m_pixels, m_pixels)
    # 1 back to array
    tmp = vec.reshape(size)
    # 2 turn the order back
    tmp[1::2] = tmp[1::2, ::-1]
    return tmp


# In[20]:


X = np.load('./data/X_256x256_v2.npy')
X_images = np.load('./data/X_images_256x256_v2.npy')
Y = np.load('./data/Y_256x256_v2.npy')





# ### Convolutional Autoencoder

# In[22]:


def create_cnn_ae(summary=True):
    #desc='128x64_conv3x3_pool2x2_relu_sigmoid'
    desc='128x64_conv3x3_pool2x2_relu_sigmoid'
    input_shape = (reduced_width,reduced_height,1)
    input_shape

    model = Sequential()
    convkernel=(3,3)
    #1st convolution layer
    model.add(Conv2D(128, convkernel #64 is number of filters and (3, 3) is the size of the filter.
        , padding='same', input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2), padding='same'))

    #2nd convolution layer
    model.add(Conv2D(64,convkernel, padding='same')) # apply 16 filters sized of (3x3)
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2), padding='same'))

    ##2nd convolution layer
#    model.add(Conv2D(2,convkernel, padding='same')) # apply 16 filters sized of (3x3)
#    model.add(Activation('relu'))
#    model.add(MaxPooling2D(pool_size=(2,2), padding='same'))

    #3rd convolution layer
    #model.add(Conv2D(128,convkernel, padding='same')) # apply 2 filters sized of (3x3)
    #model.add(Activation('relu'))
    #model.add(MaxPooling2D(pool_size=(2,2), padding='same'))

    #model.add(Flatten())
    #model.add(Dense(units = 8192, activation = 'relu'))

    #model.add(Dense(units = 6000, activation = 'relu'))

    #-------------------------
    #model.add(Dense(units = 8192, activation = 'relu'))

    #model.add(Reshape((64,64,2)))

    #4 convolution layer
    #model.add(BatchNormalization())
    #model.add(Conv2D(128,convkernel, padding='same')) # apply 2 filters sized of (3x3)
    #model.add(Activation('relu'))
    #model.add(UpSampling2D((2, 2)))

    #5 convolution layer
    #model.add(BatchNormalization())
#    model.add(Conv2D(2,convkernel, padding='same'))
#    model.add(Activation('relu'))
#    model.add(UpSampling2D((2, 2)))


    #5 convolution layer
    #model.add(BatchNormalization())
    model.add(Conv2D(64,convkernel, padding='same'))
    model.add(Activation('relu'))
    model.add(UpSampling2D((2, 2)))

    #6 convolution layer
    #model.add(BatchNormalization())
    model.add(Conv2D(128,convkernel, padding='same'))
    model.add(Activation('relu'))
    model.add(UpSampling2D((2, 2)))


    #-------------------------

    model.add(Conv2D(1,convkernel, padding='same'))
    model.add(Activation('sigmoid'))
    #model.add(Activation('relu'))

    if summary:
        model.summary()

    return model, desc


# In[23]:


#create_cnn_ae()


# ### Autoencoder

# In[24]:


from keras import layers

def create_stacked_ae(summary=True):
    desc='3000x2000x3000xreluxsigmoid'
    # This is the size of our encoded representations
    encoding_dim = 2000  # 32 floats -> compression of factor 24.5, assuming the input is 784 floats

    # This is our input image
    input_img = keras.Input(shape=(X.shape[1],))
    # "encoded" is the encoded representation of the input
    encoded = layers.Dense(3000, activation='relu')(input_img)

    encoded = layers.Dense(encoding_dim, activation='relu')(encoded)
    #encoded = layers.Dense(2000, activation='relu')(encoded)
    #encoded = layers.Dense(2000, activation='relu')(encoded)

    decoded = layers.Dense(3000, activation='relu')(encoded)
    #decoded = layers.Dense(2000, activation='relu')(encoded)
    #decoded = layers.Dense(2000, activation='relu')(decoded)


    # "decoded" is the lossy reconstruction of the input
    decoded = layers.Dense(X.shape[1], activation='sigmoid')(encoded)

    # This model maps an input to its reconstruction
    model = keras.Model(input_img, decoded)

    encoder = keras.Model(input_img, encoded)

    # This is our encoded (32-dimensional) input
    encoded_input = keras.Input(shape=(encoding_dim,))
    # Retrieve the last layer of the autoencoder model
    decoder_layer = model.layers[-1]
    # Create the decoder model
    decoder = keras.Model(encoded_input, decoder_layer(encoded_input))

    if summary:
        model.summary()
    return model, encoder, decoder, desc


# In[25]:


#create_stacked_ae()


# In[26]:


def plot_training_error(hList=None,model_type=None,ae=None,version=None,log=False,save_path=False,path=None):
    fig, axs = plt.subplots(nrows=2, ncols=5,figsize=(20,8))

    ax = axs.ravel()
    for idx,hist in enumerate(hList):
        fold=idx+1
        if log:
            ax[idx].set_yscale('log')
        ax[idx].plot(hist.history['loss'])
        ax[idx].plot(hist.history['val_loss'])
        ax[idx].set_title("Fold: "+str(fold))
        ax[idx].set_ylabel('MSE')
        ax[idx].set_xlabel('epoch')
        ax[idx].legend(['train', 'val'], loc='upper right')
    plt.suptitle(model_type+'-autoencoder-'+ae+'-V'+version,y=1.05)
    plt.tight_layout()
    if save_path:
        plt.savefig(path)


# In[27]:


def plot_RE_distribution(hList=None,model_type=None,ae=None,version=None,phase=None, metric=None,show_lines=False,bins=30,save_path=False,path=None):
    fig, axs = plt.subplots(nrows=2, ncols=5,figsize=(20,8))
    #fold=3
    ax = axs.ravel()
    for idx in range(10):
        fold=idx+1
        fold_y = np.load(os.path.join(run_folder,'fold'+str(fold),'y_'+phase+'.npy'))

        re = np.load(os.path.join(run_folder,'fold'+str(fold),phase+'_'+metric+'.npy'))
        re_mean = np.mean(re)
        re_std = np.std(re)
        p25 = np.percentile(re,25)
        p50 = np.median(re)
        p75 = np.percentile(re,75)
        xposition = [p25,p50,p75]
        #xposition = [re_mean-3*re_std, re_mean-2*re_std,re_mean+2*re_std,re_mean+3*re_std]

        re = np.load(os.path.join(run_folder,'fold'+str(fold),phase+'_'+metric+'.npy'))
        if bins is None:
            hist, bin_edges = np.histogram(re,range=(np.min(re),np.max(re)))
        else:
            hist, bin_edges = np.histogram(re,bins=bins,range=(np.min(re),np.max(re)))

        ax[idx].hist(np.load(os.path.join(run_folder,'fold'+str(fold),phase+'_'+metric+'.npy'))[np.where(fold_y == 0)],label='Normal', bins=bin_edges,alpha=0.5,color='tab:blue')
        ax[idx].hist(np.load(os.path.join(run_folder,'fold'+str(fold),phase+'_'+metric+'.npy'))[np.where(fold_y == 1)],label='TB', bins=bin_edges,alpha=0.5, color='tab:orange')
        ax[idx].set_xlabel("Reconstruction Error - "+metric)
        #ax[idx].xticks(rotation=45)
        ax[idx].tick_params(axis='x', rotation=45)
        ax[idx].legend()
        ax[idx].set_title("Fold: "+str(fold)+" "+phase)

        if show_lines:
            for xc in xposition:
                ax[idx].axvline(x=xc, color='lightskyblue', linestyle='--')

    #fig.delaxes(ax[idx+1])
    plt.suptitle(model_type+'-autoencoder-'+ae+'-V'+version,y=1.05)
    plt.tight_layout()
    if save_path:
        plt.savefig(path)
        
def plot_all_folds_RE_distribution(hList=None,model_type=None,ae=None,version=None,phase=None, metric=None,show_lines=False,bins=30,save_path=False,path=None):
    fig, axs = plt.subplots(nrows=1, ncols=1,figsize=(16,16))
    #fold=3
    #ax = axs.ravel()
    fold_list=[]
    re_list=[]

    for idx in range(10):
        fold=idx+1
        fold_list.append(np.load(os.path.join(run_folder,'fold'+str(fold),'y_'+phase+'.npy')))
        re_list.append(np.load(os.path.join(run_folder,'fold'+str(fold),phase+'_'+metric+'.npy')))
        
    fold_y = np.concatenate(fold_list,axis=0)
    re = np.concatenate(re_list,axis=0)
    
    re_mean = np.mean(re)
    re_std = np.std(re)
    p25 = np.percentile(re,25)
    p50 = np.median(re)
    p75 = np.percentile(re,75)
    xposition = [p25,p50,p75]
    #xposition = [re_mean-3*re_std, re_mean-2*re_std,re_mean+2*re_std,re_mean+3*re_std]

    if bins is None:
        hist, bin_edges = np.histogram(re,range=(np.min(re),np.max(re)))
    else:
        hist, bin_edges = np.histogram(re,bins=bins,range=(np.min(re),np.max(re)))

    axs.hist(re[np.where(fold_y == 0)],label='Normal', bins=bin_edges,alpha=0.5,color='tab:blue')
    axs.hist(re[np.where(fold_y == 1)],label='TB', bins=bin_edges,alpha=0.5, color='tab:orange')
    axs.set_xlabel("Reconstruction Error - "+metric)
    #ax[idx].xticks(rotation=45)
    axs.tick_params(axis='x', rotation=45)
    axs.legend()
    axs.set_title("Fold: "+str(fold)+" "+phase)

    if show_lines:
        for xc in xposition:
            axs.axvline(x=xc, color='lightskyblue', linestyle='--')

    #fig.delaxes(ax[idx+1])
    plt.suptitle(model_type+'-autoencoder-'+ae+'-V'+version,y=1.05)
    plt.tight_layout()
    if save_path:
        plt.savefig(path)


# In[28]:


def plot_lists(run_folder=None,fold=None,list_name=None,list_of_files=None,savefig=False):
    fig, axs = plt.subplots(nrows=8, ncols=8,figsize=(32,48))
    ax = axs.ravel()
    
    if fold is None:
        save_filename = os.path.join(run_folder,list_name+'.png')
        dirin=os.path.join(run_folder,'test')        
    else:    
        save_filename = os.path.join(run_folder,'fold'+str(fold),'fold'+str(fold)+'_'+list_name+'.png')
        dirin=os.path.join(run_folder,'fold'+str(fold),'test')
    for idx,file in enumerate(list_of_files):
        filepath=os.path.join(dirin,file)

        img = Image.open(filepath)
        img = img.convert('L')
        arr = array(img)
        if len(arr.shape) == 3:
            arr = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
        img2 = Image.fromarray(arr)
        img3 = img2.resize((reduced_width,reduced_height), Image.ANTIALIAS)
        arr3 = np.asarray(img3)
        ax[idx].imshow(arr3, cmap='gray', vmin=0, vmax=255)
        ax[idx].set_title(file.split('_tb_')[0],fontsize= 14)
        ax[idx].set_xlabel(info_dict[file.split('_tb_')[0].split('_RE_')[-1]][1]+', '+info_dict[file.split('_tb_')[0].split('_RE_')[-1]][2],fontsize= 14)

    plt.tight_layout()
    plt.suptitle("Fold "+str(fold)+" - "+list_name, y=1,fontsize= 14)
    if savefig:
        plt.savefig(save_filename)



# In[32]:


import matplotlib.image as mpimg

def generate_lists(run_folder=None,fold=None,ae=None,metric_eval=None):

    #print('1',os.path.join(run_folder,'fold'+str(fold),'test'))
    if fold is None:
        foldpath = os.path.join(run_folder)
        dirin=os.path.join(run_folder,'test')        
    else:
        foldpath=os.path.join(run_folder,'fold'+str(fold))
        dirin=os.path.join(run_folder,'fold'+str(fold),'test')

    list_of_files = sorted(os.listdir(dirin)) #+str(mse_img).replace('.','_')+'_'+'image'+str(i)+'.png')

    if metric_eval == 'mse':
        list_normal = sorted([x for x in list_of_files if '_0.png' in x])
        list_tb = sorted([x for x in list_of_files if '_1.png' in x])
    elif metric_eval == 'ssim':
        list_normal = sorted([x for x in list_of_files if '_0.png' in x])[::-1]
        list_tb = sorted([x for x in list_of_files if '_1.png' in x])[::-1]

    if ae == 'normal':
        ###Normal

        #similarity
        #NLRE
        with open(os.path.join(foldpath,'AENormal_similarity_NLRE.txt'), 'w') as f:
            for item in list_normal[:64]:
                f.write("%s\n" % item.replace('_tb_0','').replace('_tb_1','').split('_RE_')[-1])
        plot_lists(run_folder=run_folder,fold=fold,list_name='AENormal_similarity_NLRE',list_of_files=list_normal[:64],savefig=True)

        #THRE
        with open(os.path.join(foldpath,'AENormal_THRE.txt'), 'w') as f:
            for item in list_tb[::-1][:32]:
                f.write("%s\n" % item.replace('_tb_0','').replace('_tb_1','').split('_RE_')[-1])
        plot_lists(run_folder=run_folder,fold=fold,list_name='AENormal_THRE',list_of_files=list_tb[::-1][:64],savefig=True)

        #dissimilarity
        #NHRE
        with open(os.path.join(foldpath,'AENormal_dissimilarity_NHRE.txt'), 'w') as f:
            for item in list_normal[::-1][:32]:
                f.write("%s\n" % item.replace('_tb_0','').replace('_tb_1','').split('_RE_')[-1])
        plot_lists(run_folder=run_folder,fold=fold,list_name='AENormal_dissimilarity_NHRE',list_of_files=list_normal[::-1][:64],savefig=True)

        #TLRE
        with open(os.path.join(foldpath,'AENormal_dissimilarity_TLRE.txt'), 'w') as f:
            for item in list_tb[:32]:
                f.write("%s\n" % item.replace('_tb_0','').replace('_tb_1','').split('_RE_')[-1])
        plot_lists(run_folder=run_folder,fold=fold,list_name='AENormal_dissimilarity_TLRE',list_of_files=list_tb[:64],savefig=True)

    elif ae == 'TB':
        ###TB

        #similarity
        #NHRE
        with open(os.path.join(foldpath,'AETB_NHRE.txt'), 'w') as f:
            for item in list_normal[::-1][:32]:
                f.write("%s\n" % item.replace('_tb_0','').replace('_tb_1','').split('_RE_')[-1])
        plot_lists(run_folder=run_folder,fold=fold,list_name='AETB_NHRE',list_of_files=list_normal[::-1][:64],savefig=True)

        #TLRE
        with open(os.path.join(foldpath,'AETB_similarity_TLRE.txt'), 'w') as f:
            for item in list_tb[:64]:
                f.write("%s\n" % item.replace('_tb_0','').replace('_tb_1','').split('_RE_')[-1])
        plot_lists(run_folder=run_folder,fold=fold,list_name='AETB_similarity_TLRE',list_of_files=list_tb[:64],savefig=True)

        #dissimilarity
        #NLRE
        with open(os.path.join(foldpath,'AETB_dissimilarity_NLRE.txt'), 'w') as f:
            for item in list_normal[:32]:
                f.write("%s\n" % item.replace('_tb_0','').replace('_tb_1','').split('_RE_')[-1])
        plot_lists(run_folder=run_folder,fold=fold,list_name='AETB_dissimilarity_NLRE',list_of_files=list_normal[:64],savefig=True)

        #THRE
        with open(os.path.join(foldpath,'AETB_dissimilarity_THRE.txt'), 'w') as f:
            for item in list_tb[::-1][:32]:
                f.write("%s\n" % item.replace('_tb_0','').replace('_tb_1','').split('_RE_')[-1])
        plot_lists(run_folder=run_folder,fold=fold,list_name='AETB_dissimilarity_THRE',list_of_files=list_tb[::-1][:64],savefig=True)


# ## Choose parameters for training

# In[37]:


from datetime import datetime
from datetime import timedelta

now = datetime.now()-timedelta(hours = 3) # current date and time
date_time = now.strftime("%d%m%y%H%M%S")
#print("date and time:",date_time)


if serpentina:
    model_type='stacked'
    model, encoder, decoder,desc = create_stacked_ae()
else:
    model_type='cnn'
    model,desc = create_cnn_ae()

version=date_time
#version='2021_2'
run_name=model_type+'-'+ae+'-autoencoder-'+desc+'-'+metric_eval+'-'+'V'+version
run_folder='./run/'+run_name
if not os.path.exists(run_folder):
    os.mkdir(run_folder)
    
if not os.path.exists(os.path.join(run_folder,'test')):
    os.mkdir(os.path.join(run_folder,'test'))

with open(run_folder + '/model_summary.txt','w') as fh:
    # Pass the file handle in as a lambda function to make it callable
    model.summary(print_fn=lambda x: fh.write(x + '\n'))


# In[50]:


#afile = './run/cnn-normal-autoencoder-16x32x2_conv3x3_pool2x2_relu_sigmoid-mse-V040121000657/data_dict.pickle'
#bfile ='./run/cnn-normal-autoencoder-16x32x2_conv3x3_pool2x2_relu_sigmoid-ssim-V030121235553/data_dict.pickle'
#fx='fold3'
#dx='x_test'

#with open(afile, "rb") as input_file:
#    a = pickle.load(input_file)[fx][dx]

#with open(bfile, "rb") as input_file:
#    b = pickle.load(input_file)[fx][dx]

#np.array_equal(a,b)


# In[51]:

import pickle
from sklearn.model_selection import KFold

SEED = 13
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=SEED)
#kf = KFold(n_splits=10,random_state=42,shuffle=True)
#len(kf)
#print(kf)
data_dict={}
hList=[]
train_mse_fold=[]
train_ssim_fold=[]
test_mse_fold=[]
test_ssim_fold=[]
count=0

if serpentina:
    imdata = X
else:
    imdata = X_images
for train_index, test_index in kfold.split(imdata,Y):

    count+=1
    print("Fold: "+str(count))
    if not os.path.exists(os.path.join(run_folder,'fold'+str(count))):
        os.mkdir(os.path.join(run_folder,'fold'+str(count)))

    print("TRAIN:", len(train_index), "TEST:", len(test_index))
    x_train, x_test = imdata[train_index], imdata[test_index]
    y_train, y_test = Y[train_index], Y[test_index]
    name_train, name_test = [nameList[idx] for idx in train_index], [nameList[idx] for idx in test_index]

    if ae=='normal':
        x_train = x_train[np.where(y_train == 0)]
        y_train = y_train[np.where(y_train == 0)]
        name_train = [nameList[idx] for idx in np.where(y_train == 0)[0].tolist()]

        [nameList[idx] for idx in train_index]
    elif ae=='TB':
        x_train = x_train[np.where(y_train == 1)]
        y_train = y_train[np.where(y_train == 1)]
        name_train = [nameList[idx] for idx in np.where(y_train == 1)[0].tolist()]

    name_train = [x.replace('.png','') for x in name_train]
    name_test = [x.replace('.png','') for x in name_test]

    print("Y Train Dist")
    print(pd.DataFrame(y_train)[0].value_counts())
    print("Y Test Dist")
    print(pd.DataFrame(y_test)[0].value_counts())

    model=None
    if serpentina:
        print("Using Stacked-AE")
        model, encoder, decoder, desc = create_stacked_ae(summary=False)

    else:
        print("Using CNN-AE")
        model, desc = create_cnn_ae(summary=False)

        x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1) #transform 2D 28x28 matrix to 3D (28x28x1) matrix
        x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    x_train /= 255 #inputs have to be between [0, 1]
    x_test /= 255


    data_dict['fold'+str(count)] = {'x_train':x_train,
                          'x_test':x_test,
                          'y_train':y_train,
                          'y_test':y_test,
                          'name_train':name_train,
                          'name_test':name_test
                         }

    print("Training")

    model.compile(optimizer='adam', loss='mse')
    es = EarlyStopping(monitor='val_loss', verbose=2, patience=20)
    history = model.fit(x_train, x_train
    , epochs=nepochs
    , batch_size = batchsize
    , validation_data=(x_test, x_test)
    , callbacks=[es]
    , verbose=1
                    )
    hList.append(history)

    print("Checking reconstruction")

    print("Training information")

    if not os.path.exists(os.path.join(run_folder,'fold'+str(count),'train')):
        os.mkdir(os.path.join(run_folder,'fold'+str(count),'train'))

    if not os.path.exists(os.path.join(run_folder,'fold'+str(count),'train_reconstructed')):
        os.mkdir(os.path.join(run_folder,'fold'+str(count),'train_reconstructed'))

    if not os.path.exists(os.path.join(run_folder,'fold'+str(count),'train_RE')):
        os.mkdir(os.path.join(run_folder,'fold'+str(count),'train_RE'))

    model.save(os.path.join(run_folder,'fold'+str(count),'model'))

    restored_imgs = model.predict(x_train,batch_size=4,verbose=1)
    mse_list=[]
    mae_list=[]
    ssim_list=[]

    if serpentina:
        for i in range(x_train.shape[0]):
            ximg = back_to_img(x_train[i],reduced_width)
            rimg = back_to_img(restored_imgs[i],reduced_width)

            mse_img =  mean_squared_error(ximg, rimg)
            ssim_img = ssim(ximg, rimg, data_range=rimg.max() - rimg.min())
            mse_list.append(mse_img)
            ssim_list.append(ssim_img)

            if metric_eval == 'mse':
                metric_img = mse_img
            elif metric_eval == 'ssim':
                metric_img = ssim_img

            #print(1)
            plt.imsave(os.path.join(run_folder,'fold'+str(count),'train',"{:.7f}".format(float(str(metric_img))).replace('.','_')+'_RE_'+name_train[i]+'_tb_'+str(y_train[i])+'.png'),255*ximg.reshape(reduced_width,reduced_height),cmap='gray', vmin=0, vmax=255)
            #print(2)
            plt.imsave(os.path.join(run_folder,'fold'+str(count),'train_reconstructed',"{:.7f}".format(float(str(metric_img))).replace('.','_')+'_RE_'+name_train[i]+'_tb_'+str(y_train[i])+'.png'),255*rimg.reshape(reduced_width,reduced_height),cmap='gray', vmin=0, vmax=255)
            plt.imsave(os.path.join(run_folder,'fold'+str(count),'train_RE',"{:.7f}".format(float(str(metric_img))).replace('.','_')+'_RE_'+name_train[i]+'_tb_'+str(y_train[i])+'.png'),(255 - ((ximg - rimg)**2)).reshape(reduced_width,reduced_height),cmap='gray', vmin=np.min((255 - ((ximg - rimg)**2))), vmax=np.max(255 - ((ximg - rimg)**2)))

    else:
        for i in range(x_train.shape[0]):
            ximg = x_train[i].reshape(reduced_width,reduced_height)
            rimg = restored_imgs[i].reshape(reduced_width,reduced_height)

            mse_img =  mean_squared_error(ximg, rimg)
            ssim_img = ssim(ximg, rimg, data_range=rimg.max() - rimg.min())
            mse_list.append(mse_img)
            ssim_list.append(ssim_img)

            if metric_eval == 'mse':
                metric_img = mse_img
            elif metric_eval == 'ssim':
                metric_img = ssim_img

            plt.imsave(os.path.join(run_folder,'fold'+str(count),'train',"{:.7f}".format(float(str(metric_img))).replace('.','_')+'_RE_'+name_train[i]+'_tb_'+str(y_train[i])+'.png'),255*ximg.reshape(reduced_width,reduced_height),cmap='gray', vmin=0, vmax=255)
            plt.imsave(os.path.join(run_folder,'fold'+str(count),'train_reconstructed',"{:.7f}".format(float(str(metric_img))).replace('.','_')+'_RE_'+name_train[i]+'_tb_'+str(y_train[i])+'.png'),255*rimg.reshape(reduced_width,reduced_height),cmap='gray', vmin=0, vmax=255)
            plt.imsave(os.path.join(run_folder,'fold'+str(count),'train_RE',"{:.7f}".format(float(str(metric_img))).replace('.','_')+'_RE_'+name_train[i]+'_tb_'+str(y_train[i])+'.png'),(255 - ((ximg - rimg)**2)).reshape(reduced_width,reduced_height),cmap='gray', vmin=np.min((255 - ((ximg - rimg)**2))), vmax=np.max(255 - ((ximg - rimg)**2)))


    np.save(os.path.join(run_folder,'fold'+str(count),'train_ssim'),np.asarray(ssim_list))
    np.save(os.path.join(run_folder,'fold'+str(count),'train_mse'),np.asarray(mse_list))
    np.save(os.path.join(run_folder,'fold'+str(count),'y_train'),y_train)

    print(np.mean(mse_list),np.std(mse_list))
    print(np.mean(ssim_list),np.std(ssim_list))
    train_mse_fold.append(np.mean(mse_list))
    train_ssim_fold.append(np.mean(ssim_list))

    print("Test information")

    if not os.path.exists(os.path.join(run_folder,'fold'+str(count),'test')):
        os.mkdir(os.path.join(run_folder,'fold'+str(count),'test'))

    if not os.path.exists(os.path.join(run_folder,'fold'+str(count),'test_reconstructed')):
        os.mkdir(os.path.join(run_folder,'fold'+str(count),'test_reconstructed'))

    if not os.path.exists(os.path.join(run_folder,'fold'+str(count),'test_RE')):
        os.mkdir(os.path.join(run_folder,'fold'+str(count),'test_RE'))

    test_restored_imgs = model.predict(x_test,batch_size=4,verbose=1)
    test_mse_list=[]
    test_mae_list=[]
    test_ssim_list=[]

    if serpentina:
        for i in range(x_test.shape[0]):
            ximg = back_to_img(x_test[i],reduced_width)
            rimg = back_to_img(test_restored_imgs[i],reduced_width)

            mse_img =  mean_squared_error(ximg, rimg)
            ssim_img = ssim(ximg, rimg, data_range=rimg.max() - rimg.min())
            test_mse_list.append(mse_img)
            test_ssim_list.append(ssim_img)

            if metric_eval == 'mse':
                metric_img = mse_img
            elif metric_eval == 'ssim':
                metric_img = ssim_img

            plt.imsave(os.path.join(run_folder,'fold'+str(count),'test',"{:.7f}".format(float(str(metric_img))).replace('.','_')+'_RE_'+name_test[i]+'_tb_'+str(y_test[i])+'.png'),255*ximg.reshape(reduced_width,reduced_height),cmap='gray', vmin=0, vmax=255)
            plt.imsave(os.path.join(run_folder,'fold'+str(count),'test_reconstructed',"{:.7f}".format(float(str(metric_img))).replace('.','_')+'_RE_'+name_test[i]+'_tb_'+str(y_test[i])+'.png'),255*rimg.reshape(reduced_width,reduced_height),cmap='gray', vmin=0, vmax=255)
            plt.imsave(os.path.join(run_folder,'fold'+str(count),'test_RE',"{:.7f}".format(float(str(metric_img))).replace('.','_')+'_RE_'+name_test[i]+'_tb_'+str(y_test[i])+'.png'),(255 - ((ximg - rimg)**2)).reshape(reduced_width,reduced_height),cmap='gray', vmin=np.min((255 - ((ximg - rimg)**2))), vmax=np.max(255 - ((ximg - rimg)**2)))

            plt.imsave(os.path.join(run_folder,'test',"{:.7f}".format(float(str(metric_img))).replace('.','_')+'_RE_'+name_test[i]+'_tb_'+str(y_test[i])+'.png'),255*ximg.reshape(reduced_width,reduced_height),cmap='gray', vmin=0, vmax=255)
            
    else:
        for i in range(x_test.shape[0]):
            ximg = x_test[i].reshape(reduced_width,reduced_height)
            rimg = test_restored_imgs[i].reshape(reduced_width,reduced_height)

            mse_img =  mean_squared_error(ximg, rimg)
            ssim_img = ssim(ximg, rimg, data_range=rimg.max() - rimg.min())
            test_mse_list.append(mse_img)
            test_ssim_list.append(ssim_img)

            if metric_eval == 'mse':
                metric_img = mse_img
            elif metric_eval == 'ssim':
                metric_img = ssim_img

            #print(1)
            plt.imsave(os.path.join(run_folder,'fold'+str(count),'test',"{:.7f}".format(float(str(metric_img))).replace('.','_')+'_RE_'+name_test[i]+'_tb_'+str(y_test[i])+'.png'),255*ximg.reshape(reduced_width,reduced_height),cmap='gray', vmin=0, vmax=255)
            #print(2)
            plt.imsave(os.path.join(run_folder,'fold'+str(count),'test_reconstructed',"{:.7f}".format(float(str(metric_img))).replace('.','_')+'_RE_'+name_test[i]+'_tb_'+str(y_test[i])+'.png'),255*rimg.reshape(reduced_width,reduced_height),cmap='gray', vmin=0, vmax=255)
            plt.imsave(os.path.join(run_folder,'fold'+str(count),'test_RE',"{:.7f}".format(float(str(metric_img))).replace('.','_')+'_RE_'+name_test[i]+'_tb_'+str(y_test[i])+'.png'),(255 - ((ximg - rimg)**2)).reshape(reduced_width,reduced_height),cmap='gray', vmin=np.min((255 - ((ximg - rimg)**2))), vmax=np.max(255 - ((ximg - rimg)**2)))
            
            plt.imsave(os.path.join(run_folder,'test',"{:.7f}".format(float(str(metric_img))).replace('.','_')+'_RE_'+name_test[i]+'_tb_'+str(y_test[i])+'.png'),255*ximg.reshape(reduced_width,reduced_height),cmap='gray', vmin=0, vmax=255)

    np.save(os.path.join(run_folder,'fold'+str(count),'test_ssim'),np.asarray(test_ssim_list))
    np.save(os.path.join(run_folder,'fold'+str(count),'test_mse'),np.asarray(test_mse_list))
    np.save(os.path.join(run_folder,'fold'+str(count),'y_test'),y_test)
    test_mse_fold.append(np.mean(test_mse_list))
    test_ssim_fold.append(np.mean(test_ssim_list))

    generate_lists(run_folder=run_folder,fold=count,ae=ae,metric_eval=metric_eval)

with open(os.path.join(run_folder,'data_dict.pickle'), 'wb') as handle:
    pickle.dump(data_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)


with open(os.path.join(run_folder,'Metrics_Summary.txt'), 'w') as f:
    print('Train SSIM: '+str(round(np.mean(train_ssim_fold),4))+' +- '+str(round(np.std(train_ssim_fold),4)), file=f)
    print('Test SSIM: '+str(round(np.mean(test_ssim_fold),4))+' +- '+str(round(np.std(test_ssim_fold),4)), file=f)
    print('Train MSE: '+str(round(np.mean(train_mse_fold),5))+' +- '+str(round(np.std(train_mse_fold),5)), file=f)
    print('Test MSE: '+str(round(np.mean(test_mse_fold),5))+' +- '+str(round(np.std(test_mse_fold),5)), file=f)

with open('All_Models_Metrics_Summary.txt', 'a') as f:
    print(run_folder.split('/')[-1]+';'+model_type+';'+ae+';'+str(model.count_params())+';Train;SSIM;'+str(round(np.mean(train_ssim_fold),4))+';'+str(round(np.std(train_ssim_fold),4)), file=f)
    print(run_folder.split('/')[-1]+';'+model_type+';'+ae+';'+str(model.count_params())+';Test;SSIM;'+str(round(np.mean(test_ssim_fold),4))+';'+str(round(np.std(test_ssim_fold),4)), file=f)
    print(run_folder.split('/')[-1]+';'+model_type+';'+ae+';'+str(model.count_params())+';Train;MSE;'+str(round(np.mean(train_mse_fold),5))+';'+str(round(np.std(train_mse_fold),5)), file=f)
    print(run_folder.split('/')[-1]+';'+model_type+';'+ae+';'+str(model.count_params())+';Test;MSE;'+str(round(np.mean(test_mse_fold),5))+';'+str(round(np.std(test_mse_fold),5)), file=f)

plot_training_error(hList=hList,model_type=model_type,ae=ae,version=version,log=False,save_path=True,path=os.path.join(run_folder,'training_error.png'))
plot_training_error(hList=hList,model_type=model_type,ae=ae,version=version,log=True,save_path=True,path=os.path.join(run_folder,'training_error_log.png'))
plot_RE_distribution(hList=hList,model_type=model_type,ae=ae,version=version,phase='train',show_lines=True, metric='mse',bins=30,save_path=True,path=os.path.join(run_folder,'Train_MSE_RE_distribution.png'))
plot_RE_distribution(hList=hList,model_type=model_type,ae=ae,version=version,phase='test',show_lines=True, metric='mse',bins=30,save_path=True,path=os.path.join(run_folder,'Test_MSE_RE_distribution.png'))

plot_RE_distribution(hList=hList,model_type=model_type,ae=ae,version=version,phase='test',show_lines=False, metric='mse',bins=50,save_path=True,path=os.path.join(run_folder,'Test_MSE_RE_distribution_noline.png'))

try:
    plot_RE_distribution(hList=hList,model_type=model_type,ae=ae,version=version,phase='train',show_lines=True, metric='ssim',bins=30,save_path=True,path=os.path.join(run_folder,'Train_SSIM_RE_distribution.png'))
    plot_RE_distribution(hList=hList,model_type=model_type,ae=ae,version=version,phase='test',show_lines=True, metric='ssim',bins=30,save_path=True,path=os.path.join(run_folder,'Test_SSIM_RE_distribution.png'))

    plot_RE_distribution(hList=hList,model_type=model_type,ae=ae,version=version,phase='test',show_lines=False, metric='ssim',bins=50,save_path=True,path=os.path.join(run_folder,'Test_SSIM_RE_distribution_noline.png'))
# In[ ]:

except:
    print('Could not plot re distribution for SSIM')

generate_lists(run_folder=run_folder,fold=None,ae=ae,metric_eval=metric_eval)
plot_all_folds_RE_distribution(hList=hList,model_type=model_type,ae=ae,version=version,phase='test',show_lines=True, metric='mse',bins=50,save_path=True,path=os.path.join(run_folder,'Test_MSE_RE_distribution_allfolds.png'))
plot_all_folds_RE_distribution(hList=hList,model_type=model_type,ae=ae,version=version,phase='test',show_lines=False, metric='mse',bins=50,save_path=True,path=os.path.join(run_folder,'Test_MSE_RE_distribution_allfolds_noline.png'))

