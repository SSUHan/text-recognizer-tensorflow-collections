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

def attention_based_decoder_with_loss(cnn_features, labels, args, voca_c2i, is_training=True, reuse=False):
    cnn_features = tf.transpose(cnn_features, perm=[1, 0, 2])
    CNN_WIDTH = int(cnn_features.shape[1])
    CNN_CHANNELS = int(cnn_features.shape[2])

    predict = []
    confidence = []
    attention_mask = []
    loss = 0.
    
    truncated_initializer = tf.truncated_normal_initializer(stddev=0.1)
    bias_initializer = tf.constant_initializer(value=0.0)

    with tf.variable_scope("attention", reuse=reuse):
        reduced_cnn_features = tf.reshape(tf.reduce_mean(cnn_features, axis=2), [-1, CNN_WIDTH])
        Wc = tf.get_variable(name="char_embedding",
                             shape=[len(voca_c2i.keys()), args.RNN_HIDDEN_SIZE],
                            initializer=truncated_initializer,
                            trainable=is_training)
        Ws = tf.get_variable(name="state_weight",
                             shape=[args.RNN_HIDDEN_SIZE, CNN_WIDTH],
                             initializer=truncated_initializer,
                             trainable=is_training)
        Bs = tf.get_variable(name="state_bias",
                             shape=[CNN_WIDTH],
                             initializer=bias_initializer,
                             trainable=is_training)
        Wf = tf.get_variable(name="cnn_features_weight",
                             shape=[CNN_WIDTH, CNN_WIDTH],
                             initializer=truncated_initializer,
                             trainable=is_training)
        Bf = tf.get_variable(name="cnn_features_bias",
                             shape=[CNN_WIDTH],
                             initializer=bias_initializer,
                             trainable=is_training)
        Wa = tf.get_variable(name="alpha_weight",
                             shape=[CNN_WIDTH],
                             initializer=truncated_initializer,
                             trainable=is_training)
        Wi1 = tf.get_variable(name="context_weight1",
                              shape=[CNN_CHANNELS, args.RNN_HIDDEN_SIZE],
                              initializer=truncated_initializer,
                              trainable=is_training)
        Bi1 = tf.get_variable(name="context_bias1",
                              shape=[args.RNN_HIDDEN_SIZE],
                              initializer=bias_initializer,
                              trainable=is_training)
        Wi2 = tf.get_variable(name="context_weight2",
                              shape=[CNN_CHANNELS, len(voca_c2i.keys())],
                              initializer=truncated_initializer,
                              trainable=is_training)
        Bi2 = tf.get_variable(name="context_bias2",
                              shape=[len(voca_c2i.keys())],
                              initializer=bias_initializer,
                              trainable=is_training)
        Wo = tf.get_variable(name="output_weight",
                             shape=[args.RNN_HIDDEN_SIZE, len(voca_c2i.keys())],
                             initializer=truncated_initializer,
                             trainable=is_training)
        Bo = tf.get_variable(name="output_bias",
                             shape=[len(voca_c2i.keys())],
                             initializer=bias_initializer,
                             trainable=is_training)

        lstm_cell = tf.contrib.rnn.LSTMCell(args.RNN_HIDDEN_SIZE)
        lstm_state = (tf.zeros([args.BATCH_SIZE, args.RNN_HIDDEN_SIZE]),  # c_state
                      tf.zeros([args.BATCH_SIZE, args.RNN_HIDDEN_SIZE]))  # m_state

        for seq_idx in range(args.SEQ_LENGTH):
            if seq_idx == 0:
                char_embedding = tf.nn.embedding_lookup(Wc, tf.fill([args.BATCH_SIZE], voca_c2i[u'SOS']))
            else:
                tf.get_variable_scope().reuse_variables()

                def _train():
                    return tf.nn.embedding_lookup(Wc, labels[:, seq_idx - 1])

                def _test():
                    return tf.nn.embedding_lookup(Wc, predict[seq_idx - 1])

                if is_training:
                    char_embedding = _train()
                else:
                    char_embedding = _test()

            attention = tf.nn.softmax(
                Wa * tf.tanh(tf.matmul(lstm_state[0], Ws) + Bs + tf.matmul(reduced_cnn_features, Wf) + Bf))
            attention_mask.append(tf.reshape(attention, [-1, CNN_WIDTH]))
            context = tf.reduce_sum(tf.expand_dims(attention, 2) * cnn_features, 1)
            lstm_input = char_embedding + tf.matmul(context, Wi1) + Bi1
            lstm_output, lstm_state = lstm_cell(inputs=lstm_input, state=lstm_state)
            lstm_output = tf.matmul(lstm_output, Wo) + Bo + tf.matmul(context, Wi2) + Bi2
            cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=labels[:, seq_idx],
                logits=lstm_output)
            predict.append(tf.argmax(lstm_output, 1))
            v, _ = tf.nn.top_k(tf.nn.softmax(lstm_output))
            confidence.append(v[:, 0])
            loss += tf.reduce_mean(cross_entropy)
        loss /= args.SEQ_LENGTH

        predict = tf.transpose(tf.convert_to_tensor(predict), perm=[1, 0], name="predict")
        confidence = tf.reduce_mean(tf.transpose(tf.convert_to_tensor(confidence), perm=[1, 0]), axis=1,
                                    name="confidence")
        # attention_mask = tf.transpose(tf.convert_to_tensor(attention_mask), perm=[1, 0, 2], name="attention_mask")
        attention_mask = tf.convert_to_tensor(attention_mask)
        return loss, predict, confidence, attention_mask