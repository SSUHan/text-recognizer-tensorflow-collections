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

def fan_encoder(inputs, args, is_training=True, reuse=False):
    names = []
    
    # conv1
    conv1 = normal_conv2d(inputs, filters=64, kernel_size=3, padding='same', is_training=is_training,
                         reuse=reuse, activation='relu', name='1_1')
    conv1 = normal_conv2d(conv1, filters=64, kernel_size=3, padding='same', is_training=is_training,
                         reuse=reuse, activation='relu', name='1_2')
    names.append({conv1.name: str(conv1.shape)}) # 16, W/2
    
    # conv2_x
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=(2, 2), strides=(2, 2), padding='valid')
    conv2 = normal_conv2d(pool1, filters=128, kernel_size=1, padding='same', is_training=is_training,
                         reuse=reuse, activation='relu', name='2_1')
    conv2 = res_33_block(conv2, filters=128, repeats=1, is_training=is_training)
    conv2 = normal_conv2d(conv2, filters=128, kernel_size=3, padding='same', is_training=is_training,
                          reuse=reuse, activation='relu', name='2_2')
    names.append({conv2.name: str(conv2.shape)})  # shape=(?, 16, 128, 128)

    # conv3_x
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=(2, 2), strides=(2, 2), padding='valid')
    conv3 = normal_conv2d(pool2, filters=256, kernel_size=1, padding='same', is_training=is_training,
                          reuse=reuse, activation='relu', name='3_1')
    conv3 = res_33_block(conv3, filters=256, repeats=2, is_training=is_training)
    conv3 = normal_conv2d(conv3, filters=256, kernel_size=3, padding='same', is_training=is_training,
                          reuse=reuse, activation='relu', name='3_2')
    names.append({conv3.name: str(conv3.shape)})  # shape=(?, 8, 64, 256)
    
    # conv4_x
    pad4 = tf.pad(conv3, [[0,0], [0,0], [1,1], [0,0]]) # shape=(?, 8, 66, 256)
    names.append({pad4.name: str(pad4.shape)})
    pool3 = tf.layers.max_pooling2d(inputs=pad4, pool_size=(2, 2), strides=(2, 1), padding='valid') # shape=(?, 4, 65, 256)
    conv4 = normal_conv2d(pool3, filters=512, kernel_size=1, padding='same', is_training=is_training,
                          reuse=reuse, activation='relu', name='4_1')
    conv4 = res_33_block(conv4, filters=512, repeats=5, is_training=is_training)
    conv4 = normal_conv2d(conv4, filters=512, kernel_size=3, padding='same', is_training=is_training,
                          reuse=reuse, activation='relu', name='4_2')
    names.append({conv4.name: str(conv4.shape)}) # shape=(?, 4, 65, 512)
    
    # conv5_x
    conv5 = normal_conv2d(conv4, filters=512, kernel_size=1, padding='same', is_training=is_training,
                          reuse=reuse, activation='relu', name='5_1')
    conv5 = res_33_block(conv5, filters=512, repeats=3, is_training=is_training)
    conv5 = tf.pad(conv5, [[0,0], [0,0], [1,1], [0,0]]) # shape=(?, 4, 67, 512)
    names.append({conv5.name: str(conv5.shape)})
    conv5 = normal_conv2d(conv5, filters=512, kernel_size=2, padding='valid', strides=(2,1), is_training=is_training,
                          reuse=reuse, activation='relu', name='5_2') 
    names.append({conv5.name: str(conv5.shape)}) # shape=(?, 2, 66, 512)
    
    # conv6_x
    conv6 = normal_conv2d(conv5, filters=512, kernel_size=2, padding='valid', is_training=is_training,
                          reuse=reuse, activation='relu', name='6_1')
    names.append({conv6.name: str(conv6.shape)}) # shape=(?, 1, 65, 512)
    
    cnn_out = conv6
    
    return cnn_out, names
    