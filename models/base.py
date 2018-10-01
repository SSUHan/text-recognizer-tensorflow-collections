import tensorflow as tf
import common
import os, shutil

class BASE(object):
    
    model_name = "BASE"
    
    def __init__(self, args):
        self.args = args
        
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        self.sess = tf.Session(config=config)
        
        # load voca
        self.voca_i2c = list("abcdefghijklmnopqrstuvwxyz0123456789")
        tmp = self.voca_i2c[0]
        self.voca_i2c[0] = 'None'
        self.voca_i2c += [tmp, 'SOS']
        self.voca_c2i = {c: i for i, c in enumerate(self.voca_i2c)}

        self.global_step = 0

        # declare placeholders
        self.images = tf.placeholder(tf.float32,
                                     shape=(None, args.IMAGE_HEIGHT, args.IMAGE_WIDTH, args.IMAGE_CHANNELS),
                                     name='images_ph')
        self.masks = tf.placeholder(tf.float32,
                                    shape=(None, args.IMAGE_HEIGHT, args.IMAGE_WIDTH, args.IMAGE_CHANNELS),
                                    name='masks_ph')
        self.labels = tf.placeholder(tf.int32,
                                     shape=(None, args.SEQ_LENGTH),
                                     name="labels_ph")
        # for shuffle learning
        self.is_main_turn = tf.placeholder(tf.bool, name='is_main_turn')
        
        # learning rate decay policy
        self.lr_generator = common.LR_Generator(type_name=self.args.lr_method,
                                                initial_lr=self.args.lr_value,
                                                decay_rate=self.args.lr_decay,
                                                drop_step=self.args.lr_drop_step)

        self.lr = tf.placeholder(dtype=tf.float32, name='learning_rate_ph')
        self.lr_value = self.args.lr_value
        
        # summary operation list
        self.train_smy_op = []
        self.test_smy_op = []
    
    def _ready_for_train(self):
        # set lr optimizer
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            if self.args.optimizer == 'Adam':
                self.optimizer = tf.train.AdamOptimizer(self.lr)
            elif self.args.optimizer == "Adadelta":
                self.optimizer = tf.train.AdadeltaOptimizer(self.lr)
            elif self.args.optimizer == "RMSProp":
                self.optimizer = tf.train.RMSPropOptimizer(self.lr)
            elif self.args.optimizer == "SGD":
                self.optimizer = tf.train.GradientDescentOptimizer(self.lr)
            else:
                print("Cloudn't find %s optimizer" % self.args.optimizer)
                exit(1)

            self.optim = self.optimizer.minimize(self.train_loss)

        # tensorflow saver
        self.saver = tf.train.Saver(max_to_keep=10)

        # merge tf summary operations
        self.train_smy_op = tf.summary.merge(self.train_smy_op)
        self.test_smy_op = tf.summary.merge(self.test_smy_op)
        
        # restore model
        if self.args.reload:
            # restore model
            self.restore_model()

        else:
            print("removing legacy...")
            log_dir = os.path.join(self.args.LOG_DIR, self.args.model_name_with_version)
            if os.path.exists(log_dir):
                shutil.rmtree(log_dir, ignore_errors=True)
            ckpt_dir = os.path.join(self.args.CKPT_DIR, self.args.model_name_with_version)
            if os.path.exists(ckpt_dir):
                shutil.rmtree(ckpt_dir, ignore_errors=True)
            os.makedirs(ckpt_dir)
            self.step = 0
            self.sess.run(tf.global_variables_initializer())
            self.sess.run(tf.local_variables_initializer())
            
        self._print_configures()
    
    def restore_model(self):
        ckpt_dir = os.path.join(self.args.CKPT_DIR, self.args.model_name_with_version)
        ckpt = tf.train.get_checkpoint_state(ckpt_dir)
        if ckpt and ckpt.model_checkpoint_path:
            print("restoring model...")
            print(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)
            self.step = int(ckpt.model_checkpoint_path[ckpt.model_checkpoint_path.rfind("-") + 1:])
        else:
            print("without path, initialize")
            self.step = 0
            self.sess.run(tf.global_variables_initializer())
            self.sess.run(tf.local_variables_initializer())

    
    def _print_configures(self):
        # print configures
        print("=======================================================")
        print("model_name : %s" % self.args.model_name)
        print("operating_mode : %s" % self.args.operating_mode)
        print("train_continue : %s" % str(self.args.reload))
        print("optimizer : %s" % self.args.optimizer)
        print("initial_learning_rate : %0.7f" % self.args.lr_value)
        print("learning rate method : %s" % self.args.lr_method)
        print("learing_rate_decay : %0.2f" % self.args.lr_decay)
        print("learing_rate drop step : %d" % self.args.lr_drop_step)
        print("batch_size : %d" % self.args.BATCH_SIZE)
        print("rnn_seq_length : %d" % self.args.SEQ_LENGTH)
        print("rnn_hidden_size : %d" % self.args.RNN_HIDDEN_SIZE)
        print("lambda1 value : %f" % self.args.lambda1)
        print("loss type : %s" % self.args.loss)
        print("=======================================================")