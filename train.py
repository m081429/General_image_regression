from __future__ import absolute_import, division, print_function, unicode_literals

import argparse
import logging
import os
import sys
import re
import tensorflow as tf
import matplotlib.pyplot as plt
tf.keras.backend.clear_session()
from callbacks import CallBacks
from model_factory import GetModel
from preprocess import Preprocess, format_example, update_status

###############################################################################
# Input Arguments
###############################################################################

parser = argparse.ArgumentParser(description='Run a Siamese Network with a triplet loss on a folder of images.')
parser.add_argument("-t", "--image_file_train",
                    dest='image_file_train',
                    required=True,
                    help="File path ending in folders that are to be used for model training")

parser.add_argument("-v", "--image_file_validation",
                    dest='image_file_validation',
                    default=None,
                    help="File path ending in folders that are to be used for model validation")

parser.add_argument("-m", "--model-name",
                    dest='model_name',
                    default='VGG16',
                    choices=['DenseNet121',
                             'DenseNet169',
                             'DenseNet201',
                             'InceptionResNetV2',
                             'InceptionV3',
                             'MobileNet',
                             'MobileNetV2',
                             'MobileNetV3',
                             'NASNetLarge',
                             'NASNetMobile',
                             'ResNet50',
                             'ResNetRS420',
                             'VGG16',
                             'VGG19',
                             'Xception',
                             'EfficientNetV2'],
                    help="Models available from tf.keras")

parser.add_argument("-o", "--optimizer-name",
                    dest='optimizer',
                    default='Adam',
                    choices=['Adadelta',
                             'Adagrad',
                             'Adam',
                             'Adamax',
                             'Ftrl',
                             'Nadam',
                             'RMSprop',
                             'SGD'],
                    help="Optimizers from tf.keras")

parser.add_argument("-p", "--patch_size",
                    dest='patch_size',
                    help="Patch size to use for training",
                    default=256, type=int)

parser.add_argument("-l", "--log_dir",
                    dest='log_dir',
                    default='log_dir',
                    help="Place to store the tensorboard logs")

parser.add_argument("-r", "--learning-rate",
                    dest='lr',
                    help="Learning rate",
                    default=0.0001, type=float)

parser.add_argument("-L", "--loss-function",
                    dest='loss_function',
                    default='log_cosh',
                    choices=['log_cosh',
                             'MeanSquaredError',
                             'MeanAbsoluteError',
                             'MeanAbsolutePercentageError',
                             'MeanSquaredLogarithmicError'],
                    help="Loss functions from tf.keras")

parser.add_argument("-e", "--num-epochs",
                    dest='num_epochs',
                    help="Number of epochs to use for training",
                    default=10, type=int)

parser.add_argument("-b", "--batch-size",
                    dest='BATCH_SIZE',
                    help="Number of batches to use for training",
                    default=1, type=int)

parser.add_argument("-w", "--num-workers",
                    dest='NUM_WORKERS',
                    help="Number of workers to use for training",
                    default=1, type=int)

parser.add_argument("--use-multiprocessing",
                    help="Whether or not to use multiprocessing",
                    const=True, default=False, nargs='?',
                    type=bool)

parser.add_argument("-V", "--verbose",
                    dest="logLevel",
                    choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                    default="DEBUG",
                    help="Set the logging level")


parser.add_argument("--train_num_layers",
                    dest="train_num_layers",
                    default=False,
                    help="Set the logging level")

parser.add_argument("--prev_checkpoint",
                    dest="prev_checkpoint",
                    default=False,
                    help="Set the logging level")

args = parser.parse_args()

logging.basicConfig(stream=sys.stderr, level=args.logLevel,
                    format='%(name)s (%(levelname)s): %(message)s')

logger = logging.getLogger(__name__)
logger.setLevel(args.logLevel)



        
###############################################################################
# Begin priming the data generation pipeline
###############################################################################

# Get Training and Validation data
train_data = Preprocess(args.image_file_train, loss_function=args.loss_function)
logger.debug('Completed  training dataset Preprocess')

#print(train_data.files[0:5],train_data.labels[0:5])
#sys.exit(0)
# AUTOTUNE = tf.data.experimental.AUTOTUNE
AUTOTUNE = 1000

# Update status to Training for map function in the preprocess
update_status(True)

#creating train dataset
t_path_ds = tf.data.Dataset.from_tensor_slices(train_data.files)
t_image_ds = t_path_ds.map(format_example, num_parallel_calls=AUTOTUNE)
t_label_ds = tf.data.Dataset.from_tensor_slices(tf.cast(train_data.labels, tf.float32))
t_image_label_ds = tf.data.Dataset.zip((t_image_ds, t_label_ds))
#train_ds = t_image_label_ds.shuffle(buffer_size=train_data.min_images).repeat()
train_ds = t_image_label_ds.repeat()

#creating tensorflow batch dataset
train_ds = train_ds.batch(args.BATCH_SIZE).prefetch(buffer_size=AUTOTUNE)
training_steps = int(train_data.min_images / args.BATCH_SIZE)
logger.debug('Completed Training dataset')

#if validation data is provided
if args.image_file_validation:
    # Get Validation data
    # Update status to Testing for map function in the preprocess
    update_status(False)
    #creating val dataset
    validation_data = Preprocess(args.image_file_validation, loss_function=args.loss_function)
    logger.debug('Completed test dataset Preprocess')

    v_path_ds = tf.data.Dataset.from_tensor_slices(validation_data.files)
    v_image_ds = v_path_ds.map(format_example, num_parallel_calls=AUTOTUNE)
    v_label_ds = tf.data.Dataset.from_tensor_slices(tf.cast(validation_data.labels, tf.int64))
    v_image_label_ds = tf.data.Dataset.zip((v_image_ds, v_label_ds))
   
    validation_ds = v_image_label_ds.shuffle(buffer_size=validation_data.min_images).repeat()
    validation_ds = validation_ds.batch(args.BATCH_SIZE).prefetch(buffer_size=AUTOTUNE)
    validation_steps = int(validation_data.min_images / args.BATCH_SIZE)
    logger.debug('Completed Validation dataset')

else:
    validation_ds = None
    validation_steps = None

#creating output directory
out_dir = os.path.join(args.log_dir, args.model_name + '_' + args.optimizer + '_' + str(args.lr) + '-' + args.loss_function)
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

#creating checkpoint directory
checkpoint_path = os.path.join(out_dir, "cp-{epoch:04d}.ckpt")
checkpoint_dir = os.path.dirname(checkpoint_path)

###############################################################################
# Build the model
###############################################################################

logger.debug('Mirror initialized')
# multi-GPU
mirrored_strategy = tf.distribute.MirroredStrategy()
with mirrored_strategy.scope():
    #getting model
    if args.train_num_layers:
        m = GetModel(model_name=args.model_name, img_size=args.patch_size,optimizer=args.optimizer, lr=args.lr, loss_name=args.loss_function, num_layers=int(args.train_num_layers))
    else:
        m = GetModel(model_name=args.model_name, img_size=args.patch_size,optimizer=args.optimizer, lr=args.lr, loss_name=args.loss_function )
    # compiling model
    model, preprocess = m._get_model_and_preprocess()
    model = m.compile_model(model)

    logger.debug('Model compiled')
    
    #getting latest model is output directory already exists
    latest = tf.train.latest_checkpoint(checkpoint_dir)
    if not latest:
        if args.prev_checkpoint:
            model.load_weights(args.prev_checkpoint)
            logger.debug('Loading weights from ' + args.prev_checkpoint)
        model.save_weights(checkpoint_path.format(epoch=0))
        latest = tf.train.latest_checkpoint(checkpoint_dir)
    
    #getting the initial epoch    
    ini_epoch = int(re.findall(r'\b\d+\b', os.path.basename(latest))[0])
    logger.debug('Loading initialized model')
    model.load_weights(latest)
    logger.debug('Loading weights from ' + latest)

# #automatic LR finder
# automatic_lr=1

# '''LRFinder class'''

# class LRFind(tf.keras.callbacks.Callback): 
    # def __init__(self, min_lr, max_lr, n_rounds): 
        # self.min_lr = min_lr
        # self.max_lr = max_lr
        # self.step_up = (max_lr / min_lr) ** (1 / n_rounds)
        # self.lrs = []
        # self.losses = []
     
    # def on_train_begin(self, logs=None):
        # self.weights = self.model.get_weights()
        # self.model.optimizer.lr = self.min_lr

    # def on_train_batch_end(self, batch, logs=None):
        # self.lrs.append(self.model.optimizer.lr.numpy())
        # self.losses.append(logs["loss"])
        # self.model.optimizer.lr = self.model.optimizer.lr * self.step_up
        # if self.model.optimizer.lr > self.max_lr:
            # self.model.stop_training = True
        
    # def on_train_end(self, logs=None):
        # self.model.set_weights(self.weights)

# if automatic_lr==0:
    # EPOCHS=1
    # lr_finder_steps = training_steps
    # lr_find = LRFind(1e-9, 1e1, lr_finder_steps)

    # history = model.fit(train_ds,
          # steps_per_epoch=training_steps,
          # epochs=EPOCHS,
          # callbacks=[lr_find],
          # validation_data=validation_ds,
          # validation_steps=validation_steps,
          # class_weight=None,
          # max_queue_size=1000,
          # workers=args.NUM_WORKERS,
          # use_multiprocessing=args.use_multiprocessing,
          # shuffle=False, initial_epoch=ini_epoch
          # )
    # plt.figure()      
    # plt.plot(lr_find.lrs, lr_find.losses)
    # plt.xscale('log')
    # plt.title("AUTO LRFINDER " )
    # plt.ylabel("loss", fontsize="large")
    # plt.xlabel("log LR", fontsize="large")
    # plt.savefig(os.path.join(out_dir +'LR_finder.pdf')) 
    # plt.show()
    # plt.close()
    # sys.exit(0)
# #automatic LR finder stop

logger.debug('Completed loading initialized model')
cb = CallBacks(learning_rate=args.lr, log_dir=out_dir, optimizer=args.optimizer)
logger.debug('Model image saved')

#running the model
history = model.fit(train_ds,
          steps_per_epoch=training_steps,
          epochs=args.num_epochs,
          callbacks=cb.get_callbacks(),
          validation_data=validation_ds,
          validation_steps=validation_steps,
          class_weight=None,
          max_queue_size=1000,
          workers=args.NUM_WORKERS,
          use_multiprocessing=args.use_multiprocessing,
          shuffle=False, initial_epoch=ini_epoch
          )
model.save(os.path.join(out_dir, 'my_model.h5'))

#plotting 
metric = "mae"
plt.figure()
plt.plot(history.history[metric])
plt.plot(history.history["val_" + metric])
plt.title("model " + metric)
plt.ylabel(metric, fontsize="large")
plt.xlabel("epoch", fontsize="large")
plt.legend(["train", "val"], loc="best")
plt.savefig(os.path.join(out_dir, metric+'.pdf')) 
plt.show()
plt.close()


metric = "loss"
plt.figure()
plt.plot(history.history[metric])
plt.plot(history.history["val_" + metric])
plt.title("model " + metric)
plt.ylabel(metric, fontsize="large")
plt.xlabel("epoch", fontsize="large")
plt.legend(["train", "val"], loc="best")
plt.savefig(os.path.join(out_dir, metric+'.pdf')) 
plt.show()
plt.close() 