import common
import tensorflow as tf
from models.CRNN import CRNN

class TextRecognizer:
    
    def __init__(self, args):
        self.args = args

if __name__ == "__main__":
    args = common.parse_args()
    recog_model = CRNN(args=args)
    recog_model.train()
    
    exit(0)
    
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
    
    