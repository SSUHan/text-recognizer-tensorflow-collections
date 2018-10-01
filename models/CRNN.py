import tensorflow as tf
from models.base import BASE
from models.encoder_backbone import crnn_encoder
from models.decoder_backbone import normal_bilstm, ctc_based_decoder
from models.data_loader import RecognizeDataLoader
from pprint import pprint

from timeit import default_timer as timer

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
        
    def build_model(self, scope, is_debug=False):
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
        self.train_loss, self.train_predict = self.build_model(scope="train", is_debug=True)
        self.test_loss, self.test_predict = self.build_model(scope="test", is_debug=False)
        super()._ready_for_train()
        
        # train data loader
        self.train_recognizer_data = RecognizeDataLoader(voca_i2c=self.voca_i2c, voca_c2i=self.voca_c2i, 
                                                         data_dir='train_data/data_recognition_11.6M', pickle_name='train', args=self.args)
        
        train_loss = 0.
        train_accuracy = 0.
        self.unchanged_count = 0
        self.best_valid_loss = 0.
        self.best_valid_accuracy= 0.
        self.epoch = int(self.step / self.train_recognizer_data.size())
        
        # loop
        while self.epoch < self.args.MAX_EPOCHS:
            start_time = timer()
            batch = self.train_recognizer_data.get_batch()
            if batch is None:
                self.epoch += 1
                continue
            
            if self.lr_value < self.args.MIN_LEARNING_RATE:
                print("Stop minimum learning rate value..")
                break
            
            print("Epoch : [ {} ], step : [ {} ], get_batch time : [{0:.2f}], ".format(self.epoch, self.step, timer() - start_time), end='')
            images, masks, labels = batch
            
            # get lr value from lr_generator
            self.lr_value, is_unchange_decay = self.lr_generator.get_lr(current_lr=self.lr_value,
                                                                       current_step=self.step,
                                                                       unchanged_count=self.unchanged_count)
            if is_unchange_decay:
                self.unchanged_count = 0
            
            # feed to session
            input_values = [self.optim, self.train_loss, self.train_predict, self.train_smy_op]
            feed_dict = {
                self.images: images, 
                self.labels: labels,
                self.lr: self.lr_value
            }
            
            # update phrase
            start_time = timer()
            _, _loss, _predict, _summaries = self.sess.run(input_values, feed_dict=feed_dict)
            print("update time : [{0:.2f}], train_loss : [ {0:.5f} ]".format(timer() - start_time, _loss))
            train_loss += _loss
            correct = super()._get_correct_predict(_predict, labels)
            train_accuracy += 100*float(correct) / float(_predict.shape[0])
            
            # write train scope log
            if self.step != 0 and self.step % self.args.LOG_INTERVAL == 0:
                self.train_loss_value = float(train_loss / self.args.LOG_INTERVAL)
                self.train_accuracy_value = float(train_accuracy / self.args.LOG_INTERVAL)
                super()._print_train_interval()
                
                self.train_writer.add_summary(tf.Summary(
                value=[
                    tf.Summary.Value(tag='train/recognizer_accuracy', simple_value=self.train_accuracy_value),
                    tf.Summary.Value(tag='learning_rate', simple_value=self.lr_value)
                ]), self.step)
                self.train_writer.add_summary(_summaries, self.step)
                train_loss = 0.
                train_accuracy = 0.
            
            # write valid scope log
            if self.step != 0 and self.step % self.args.VALID_INTERVAL == 0:
                pass
            
            self.step += 1
            
            
        