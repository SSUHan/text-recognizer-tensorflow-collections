import tensorflow as tf
from models.layers import *

def normal_bilstm(cnn_out, args, is_training=True, reuse=False):
    names = []
    CNN_WIDTH = cnn_out.shape[2]
    CNN_CHANNELS = cnn_out.shape[3]
    cnn_features = tf.reshape(cnn_out, shape=[-1, CNN_WIDTH, CNN_CHANNELS])
    names.append({cnn_features.name: str(cnn_features.shape)})
    cnn_features = tf.transpose(cnn_features, perm=[1, 0, 2])
    names.append({cnn_features.name: str(cnn_features.shape)})
    for stack_idx in range(args.RNN_STACK_SIZE):
        cnn_features = BiLSTM(cnn_features, int(args.RNN_HIDDEN_SIZE / 2), name='bilstm_%d' % stack_idx)
        names.append({cnn_features.name: str(cnn_features.shape)})
    return cnn_features, names

def ctc_based_decoder(input_features, args, voca_c2i, is_training, reuse=False):
    
    seq_length = tf.fill([args.BATCH_SIZE], int(input_features.shape[0]))
    logits = tf.layers.dense(input_features, len(voca_c2i.keys())+ 1,
                             kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                             trainable=is_training,
                             name='ctc_logits')
    predict, confidence = tf.nn.ctc_greedy_decoder(logits, seq_length)
    predict = tf.sparse_tensor_to_dense(predict[0], name='predict')
    confidence = tf.reduce_mean(confidence, axis=1, name='confidence')

    return logits, predict, seq_length, confidence