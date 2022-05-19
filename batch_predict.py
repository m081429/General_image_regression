
from __future__ import absolute_import, division, print_function, unicode_literals
import argparse
import logging
import os
import sys
import tensorflow as tf
import matplotlib.pylab as plt
from tensorflow.keras import layers
from tensorflow.keras import models
import numpy as np
import PIL.Image as Image
import glob
from preprocess import Preprocess, format_example,  update_status
from model_factory import GetModel

#input params
input_file_path=sys.argv[1]
input_file_path=input_file_path.strip()
model_path=sys.argv[2]
model_path=model_path.strip()
output_file=sys.argv[3]
output_file=output_file.strip()
BATCH_SIZE=64
IMAGE_SHAPE = (256, 256)




'''Loadmodel using hdf5 file'''
#new_model = models.load_model(model_path)

'''Alternate way to create the model and run'''
model_name=os.path.basename(os.path.basename(model_path))

#getting model name, optimizer, loss function, lr from the directory name
model_type = model_name.split("_")[0]
optimizer = model_name.split("_")[1]
loss_function = model_name.split("-")[-1]
lr = model_name.replace('-'+loss_function,"").split("_")[2]

'''Loadmodel'''
print(model_name,IMAGE_SHAPE[0],optimizer,lr,loss_function)

#creating model
m = GetModel(model_name=model_type, img_size=IMAGE_SHAPE[0],optimizer=optimizer, lr=lr, loss_name=loss_function)
model, preprocess = m._get_model_and_preprocess()
#compiling model
model = m.compile_model(model)
#getting latest checkpoint file and loading the model with latest weights
latest = tf.train.latest_checkpoint(model_path)
model.load_weights(latest)

'''reading the file paths'''
files_list=[]
labels_list=[]
fobj = open(input_file_path)
for x in fobj:
    x=x.strip()
    arr=x.split(" ")
    files_list.append(arr[0])
    labels_list.append(float(arr[1])*100)
fobj.close()   

#this is prediction,so no need of additional augmentation 
update_status(False)
len_files_list = len(files_list)

#creating dataset
t_path_ds = tf.data.Dataset.from_tensor_slices(files_list)
t_image_ds = t_path_ds.map(format_example, num_parallel_calls=tf.data.experimental.AUTOTUNE)
t_label_ds = tf.data.Dataset.from_tensor_slices(tf.cast(labels_list, tf.float32))
t_path_ds = tf.data.Dataset.from_tensor_slices(files_list)
t_image_label_ds = tf.data.Dataset.zip((t_image_ds, t_label_ds,t_path_ds))
train_ds = t_image_label_ds.batch(BATCH_SIZE).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
training_steps = int(len_files_list / BATCH_SIZE)

#multi gpu
strategy = tf.distribute.MirroredStrategy()
fobj_w = open(output_file,'w')
with strategy.scope():
    #iterating through each batch
    for step, (image,label,file) in enumerate(train_ds):
        #predicting
        lst_label = list(label.numpy())
        lst_file = list(file.numpy())
        result = np.asarray(model.predict_on_batch(image))
        lst_result = list(result)
        for i in range(0,len(lst_label)):
            #writing out
            fobj_w.write(model_name+"\t"+lst_file[i].decode("utf-8")+"\t"+str(lst_label[i])+"\t"+str(lst_result[i][0])+"\n")
fobj_w.close()            