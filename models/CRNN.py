import tensorflow as tf
from models.base import BASE
from models.encoder_backbone import crnn_encoder
from models.decoder_backbone import normal_bilstm, ctc_based_decoder
from pprint import pprint

class CRNN(BASE):
    
    model_name = "CRNN"
    
    def __del__(self):
        self.sess.close()
    
    def __init__(self, args):
        super().__init__(args)
        self.layers = []
    
    def encoder(self, is_training=True, reuse=False):
        with tf.variable_scope('encoder', reuse=reuse):
            cnn_out, names = crnn_encoder(self.images, self.args, is_training=is_training, reuse=reuse)
        return cnn_out, names
    
    def decoder(self, cnn_out, is_training=True, reuse=False):
        with tf.variable_scope('rnn_decoder', reuse=reuse):
            rnn_out, names = normal_bilstm(cnn_out, self.args, is_training=is_training, reuse=reuse)
        
        with tf.variable_scope('ctc_decoder', reuse=reuse):
            logits, predict, seq_length, confidence = ctc_based_decoder(rnn_out, self.args, self.voca_c2i,
                                                                        is_training=is_training, reuse=reuse)
        
        return logits, predict, seq_length, confidence, names
        
    def get_loss(self, logits, labels, seq_length, scope="train"):
        # convert labels from dense to sparse
        indices = tf.where(tf.not_equal(labels, self.voca_c2i[u'None']))
        labels = tf.reshape(labels, [self.args.BATCH_SIZE, self.args.SEQ_LENGTH])
        sparse_labels = tf.SparseTensor(indices, tf.gather_nd(labels, indices), labels.get_shape())

        ctc_loss = tf.reduce_mean(
            tf.nn.ctc_loss(
                inputs=logits,
                labels=sparse_labels,
                sequence_length=seq_length
            )
        )
        
        if scope == "train":
            self.train_smy_op.append(tf.summary.scalar("{}/ctc_loss".format(scope), ctc_loss))
        elif scope == "test":
            self.test_smy_op.append(tf.summary.scalar("{}/ctc_loss".format(scope), ctc_loss))
        
        return ctc_loss
        
    def build_model(self, scope="train", is_debug=False):
        if scope == "train":
            cnn_out, names = self.encoder(is_training=True, reuse=False)
            self.layers.extend(names)
            logits, predict, seq_length, confidence, names = self.decoder(cnn_out, is_training=True, reuse=False)
            self.layers.extend(names)
            loss = self.get_loss(logits, self.labels, seq_length, scope=scope)
            
            if is_debug:
                pprint(self.layers)
            
            return loss, predict
        
        elif scope == "test":
            cnn_out, names = self.encoder(is_training=False, reuse=True)
            logits, predict, seq_length, confidence, names = self.decoder(cnn_out, is_training=False, reuse=True)
            loss = self.get_loss(logits, self.labels, seq_length, scope=scope)
            return loss, predict
        
        elif scope == "infer":
            cnn_out, names = self.encoder(is_training=False, reuse=False)
            logits, predict, seq_length, confidence, names = self.decoder(cnn_out, is_training=False, reuse=False)
            return None, predict
    
    def train(self):
        pass
        
        