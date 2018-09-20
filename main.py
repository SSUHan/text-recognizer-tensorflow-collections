import common
import tensorflow as tf

class TextRecognizer():
    
    def __del__(self):
        self.sess.close()
    
    def __init__(self, args):
        self.args = args
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        self.sess = tf.Session(config=config)

if __name__ == "__main__":
    args = common.parse_args()
    
    with tf.device('/gpu:{}'.format(args.gpu)):
        ocr_recognizer = TextRecognizer(args)
        if args.operating_mode == "train":
            pass
        elif args.operating_mode == 'test':
            pass
        elif args.operating_mode == 'infer':
            pass
        pass
    
    common.args2json(args)
    
    