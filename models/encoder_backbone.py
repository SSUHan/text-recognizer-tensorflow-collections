import tensorflow as tf
from models.layers import *

def crnn_encoder(inputs, args, is_training=True, reuse=False):
    names = []
    # conv1 
    conv1 = normal_conv2d(inputs, filters=64, kernel_size=3, padding='same', is_training=is_training,
                         reuse=reuse, activation='relu', name='1_1')
    names.append({conv1.name: str(conv1.shape)}) # 32, W
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=(2, 2), strides=(2, 2), padding='valid')
    names.append({pool1.name: str(pool1.shape)}) # 16, W/2
    
    # conv2
    conv2 = normal_conv2d(inputs=pool1, filters=128, kernel_size=3, padding='same', is_training=is_training,
                         reuse=reuse, activation='relu', name='2_1')
    names.append({conv2.name: str(conv2.shape)}) # 16, W/2
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=(2, 2), strides=(2, 2), padding='valid')
    names.append({pool2.name: str(pool2.shape)}) # 8, W/4
    
    # conv3
    conv3 = normal_conv2d(inputs=pool2, filters=256, kernel_size=3, padding='same', is_training=is_training,
                         reuse=reuse, activation='relu', name='3_1')
    names.append({conv3.name: str(conv3.shape)}) # 8, W/4
    conv3 = normal_conv2d(inputs=conv3, filters=256, kernel_size=3, padding='same', is_training=is_training,
                         reuse=reuse, activation='relu', name='3_2')
    names.append({conv3.name: str(conv3.shape)}) # 8, W/4
    pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=(2, 1), strides=(2, 1), padding='valid')
    names.append({pool3.name: str(pool3.shape)}) # 4, W/4
    
    # conv4
    conv4 = normal_conv2d(inputs=pool3, filters=512, kernel_size=3, padding='same', is_training=is_training,
                         reuse=reuse, activation='relu', name='4_1', batch_norm=True)
    names.append({conv4.name: str(conv4.shape)}) # 4, W/4
    conv4 = normal_conv2d(inputs=conv4, filters=512, kernel_size=3, padding='same', is_training=is_training,
                         reuse=reuse, activation='relu', name='4_2', batch_norm=True)
    names.append({conv4.name: str(conv4.shape)}) # 4, W/4
    pool4 = tf.layers.max_pooling2d(inputs=conv4, pool_size=(2, 1), strides=(2, 1), padding='valid')
    names.append({pool4.name: str(pool4.shape)}) # 2, W/4
    
    # conv5
    conv5 = normal_conv2d(inputs=pool4, filters=512, kernel_size=2, padding='valid', is_training=is_training,
                         reuse=reuse, activation='relu', name='5_1')
    names.append({conv5.name: str(conv5.shape)}) # 1, W/4-1
    
    cnn_out = conv5
    return cnn_out, names
    