import tensorflow as tf

def BiLSTM(features, rnn_size, name):
    with tf.variable_scope(name):
        cell_fw = tf.contrib.rnn.LSTMCell(rnn_size,
                                          initializer=tf.truncated_normal_initializer(stddev=0.01))
        cell_bw = tf.contrib.rnn.LSTMCell(rnn_size,
                                          initializer=tf.truncated_normal_initializer(stddev=0.01))
        output, _ = tf.nn.bidirectional_dynamic_rnn(
            cell_fw,
            cell_bw,
            inputs=features,
            dtype=tf.float32,
            time_major=True
        )
        return tf.concat(output, axis=2)

def ctc_based_decoder(input_features, voca_c2i, is_training, reuse=False, params=None):
    with tf.variable_scope("CTC", reuse=reuse):
        seq_length = tf.fill([params.BATCH_SIZE], int(input_features.shape[0]))
        logits = tf.layers.dense(input_features, len(voca_c2i.keys())+ 1,
                                 kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                                 trainable=is_training,
                                 name='ctc_logits')
        predict, confidence = tf.nn.ctc_greedy_decoder(logits, seq_length)
        predict = tf.sparse_tensor_to_dense(predict[0], name='predict')
        confidence = tf.reduce_mean(confidence, axis=1, name='confidence')

        return logits, predict, seq_length, confidence


def attention_based_decoder_with_loss(cnn_features, labels, voca_c2i, is_training, reuse=False, params=None):
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
                             shape=[len(voca_c2i.keys()), params.RNN_HIDDEN_SIZE],
                            initializer=truncated_initializer,
                            trainable=is_training)
        Ws = tf.get_variable(name="state_weight",
                             shape=[params.RNN_HIDDEN_SIZE, CNN_WIDTH],
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
                              shape=[CNN_CHANNELS, params.RNN_HIDDEN_SIZE],
                              initializer=truncated_initializer,
                              trainable=is_training)
        Bi1 = tf.get_variable(name="context_bias1",
                              shape=[params.RNN_HIDDEN_SIZE],
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
                             shape=[params.RNN_HIDDEN_SIZE, len(voca_c2i.keys())],
                             initializer=truncated_initializer,
                             trainable=is_training)
        Bo = tf.get_variable(name="output_bias",
                             shape=[len(voca_c2i.keys())],
                             initializer=bias_initializer,
                             trainable=is_training)

        lstm_cell = tf.contrib.rnn.LSTMCell(params.RNN_HIDDEN_SIZE)
        lstm_state = (tf.zeros([params.BATCH_SIZE, params.RNN_HIDDEN_SIZE]),  # c_state
                      tf.zeros([params.BATCH_SIZE, params.RNN_HIDDEN_SIZE]))  # m_state

        for seq_idx in range(params.SEQ_LENGTH):
            if seq_idx == 0:
                char_embedding = tf.nn.embedding_lookup(Wc, tf.fill([params.BATCH_SIZE], voca_c2i[u'SOS']))
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
        loss /= params.SEQ_LENGTH

        predict = tf.transpose(tf.convert_to_tensor(predict), perm=[1, 0], name="predict")
        confidence = tf.reduce_mean(tf.transpose(tf.convert_to_tensor(confidence), perm=[1, 0]), axis=1,
                                    name="confidence")
        # attention_mask = tf.transpose(tf.convert_to_tensor(attention_mask), perm=[1, 0, 2], name="attention_mask")
        attention_mask = tf.convert_to_tensor(attention_mask)
        return loss, predict, confidence, attention_mask

def legacy_attention_decoder(cnn_features, labels, voca_c2i, is_training, reuse=False, params=None):
    weight_initializer = tf.contrib.layers.variance_scaling_initializer()
    bias_initializer = tf.constant_initializer(value=0.0)

    attention_states = tf.transpose(cnn_features, perm=[1, 0, 2])
    BATCH_SIZE = int(cnn_features.shape[0])
    CNN_WIDTH = int(cnn_features.shape[1])
    CNN_CHANNELS = int(cnn_features.shape[2])

    with tf.variable_scope('attention'):
        # Make decoder inputs(dummy)
        dummy_label = tf.concat([tf.zeros([BATCH_SIZE, len(voca_c2i.keys())]),
                                 tf.ones([BATCH_SIZE, 1])],
                                axis=-1) # [BATCH_SIZE, num_classes + 1]
        decoder_inputs = [dummy_label] + [None] * (params.SEQ_LENGTH - 1)




def ctc_based_decoder_with_loss(input_features, labels, voca_c2i, is_training, reuse=False, params=None):
    with tf.variable_scope("CTC", reuse=reuse):
        seq_length = tf.fill([params.BATCH_SIZE], int(input_features.shape[0]))
        logits = tf.layers.dense(input_features, len(voca_c2i.keys())+ 1,
                                 kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                                 trainable=is_training,
                                 name='ctc_logits')
        predict, confidnece = tf.nn.ctc_greedy_decoder(logits, seq_length)

        predict = tf.sparse_tensor_to_dense(predict[0], name="predict")
        confidnece = tf.reduce_mean(confidnece, axis=1, name="confidence")

        # convert labels from dense to sparse
        indices = tf.where(tf.not_equal(labels, voca_c2i[u'None']))
        labels = tf.reshape(labels, [params.BATCH_SIZE, params.SEQ_LENGTH])
        sparse_labels = tf.SparseTensor(indices, tf.gather_nd(labels, indices), labels.get_shape())

        loss = tf.reduce_mean(
            tf.nn.ctc_loss(
                inputs=logits,
                labels=sparse_labels,
                sequence_length=seq_length
            )
        )

        return loss, predict, confidnece

def normal_conv2d(inputs, filters, padding, kernel_size, strides=(1,1), name='1', is_training=True, activation='relu', reuse=False, batch_norm=False):
    if activation == 'relu':
        acti_func = tf.nn.relu
    elif activation == 'sigmoid':
        acti_func = tf.nn.sigmoid

    with tf.variable_scope("normal_conv2d_{}".format(name), reuse=reuse):
        conv = tf.layers.conv2d(inputs=inputs, filters=filters, padding=padding, kernel_size=kernel_size, strides=strides, name="conv")
        if batch_norm:
            conv = acti_func(tf.layers.batch_normalization(inputs=conv, training=is_training, name="bn"))
        else:
            conv = acti_func(conv)
    return conv

def res_33_block(inputs, filters, repeats, is_training):
    h = inputs
    for i in range(repeats):
        if h.shape[-1] == filters:
            start = h
        else:
            start = tf.layers.conv2d(inputs=h, filters=filters, padding='same', kernel_size=1)
        h = tf.layers.conv2d(inputs=h, filters=filters, padding='same', kernel_size=3)
        h = tf.layers.batch_normalization(inputs=h, training=is_training)
        h = tf.nn.relu(h)
        h = tf.layers.conv2d(inputs=h, filters=filters, padding='same', kernel_size=3)
        h = tf.layers.batch_normalization(inputs=h, training=is_training)
        h = tf.nn.relu(start + h)  # residual connection
    return h

def res_131_block(inputs, filters, strides, repeats, is_training):
    h = inputs

    for i in range(repeats):
        # BN-ReLU 1x1 Conv
        if i == 0:
            h = tf.layers.batch_normalization(inputs=h, training=is_training)
            h = tf.nn.relu(h)
            h = tf.layers.conv2d(inputs=h, filters=filters, padding='same', strides=strides, kernel_size=1)
            start = h
        else:
            start = h
            h = tf.layers.batch_normalization(inputs=h, training=is_training)
            h = tf.nn.relu(h)
            h = tf.layers.conv2d(inputs=h, filters=filters, padding='same', kernel_size=1)

        # BN-ReLU 3x3 Conv
        h = tf.layers.batch_normalization(inputs=h, training=is_training)
        h = tf.nn.relu(h)
        h = tf.layers.conv2d(inputs=h, filters=filters, padding='same', kernel_size=3)

        # BN-ReLU 1x1 Conv
        h = tf.layers.batch_normalization(inputs=h, training=is_training)
        h = tf.nn.relu(h)
        h = tf.layers.conv2d(inputs=h, filters=filters * 4, padding='same', kernel_size=1)

        if start.shape[3] != h.shape[3]:
            start = tf.layers.batch_normalization(inputs=start, training=is_training)
            start = tf.nn.relu(start)
            start = tf.layers.conv2d(inputs=start, filters=filters * 4, padding='same', kernel_size=1)
        h = start + h
    return h

def deconv2d(_input, output_shape, kernel_hw, stride_hw, padding='SAME', name='deconv2d', stddev=0.02, with_w=False):
    kernel_height, kernel_width = kernel_hw
    stride_height, stride_width = stride_hw
    with tf.variable_scope(name):
        w = tf.get_variable('w', [kernel_height, kernel_width, output_shape[-1], _input.get_shape()[-1]],
                            initializer=tf.random_normal_initializer(stddev=stddev))

        try:
            deconv = tf.nn.conv2d_transpose(_input, w, output_shape=output_shape, strides=[1, stride_height, stride_width, 1], padding=padding)

        # Support for versions of Tensorflow befor 0.7.0
        except AttributeError:
            deconv = tf.nn.deconv2d(_input, w, output_shape=output_shape, strides=[1, stride_height, stride_width, 1])

        biases = tf.get_variable('biases', [output_shape[-1]], initializer=tf.constant_initializer(0.0))
        deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())

        if with_w:
            return deconv, w, biases
        else:
            return deconv

def batch_norm(x, is_training, scope):
    return tf.contrib.layers.batch_norm(x,
                                        decay=0.9,
                                        updates_collections=None,
                                        epsilon=1e-5,
                                        scale=True,
                                        is_training=is_training,
                                        scope=scope)


