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


