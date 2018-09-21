import tensorflow as tf

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
        self.is_main_turn = tf.placeholder(tf.bool, name='is_main_turn')
        
        self.train_smy_op = []
        self.test_smy_op = []