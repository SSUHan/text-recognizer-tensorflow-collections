import common
import tensorflow as tf
from models.CRNN import CRNN
from models.FAN import FAN

class TextRecognizer:
    
    def __init__(self, args):
        self.args = args
        
        self.recog_model = None
        if self.args.model_name == "CRNN":
            self.recog_model = CRNN(args=args)
        elif self.args.model_name == "FAN":
            self.recog_model = FAN(args=args)
        else:
            print("Unknown model name for initialize.. {}".format(self.args.model_name))
            exit(0)
    
    def build_model(self, scope='train', is_debug=True):
        return self.recog_model.build_model(scope=scope, is_debug=is_debug)
    
    def train(self):
        return self.recog_model.train()

if __name__ == "__main__":
    args = common.parse_args()    
    with tf.device('/gpu:{}'.format(args.gpu)):
        ocr_recognizer = TextRecognizer(args)
        if args.operating_mode == "train":
            ocr_recognizer.train()
        elif args.operating_mode == 'test':
            pass
        elif args.operating_mode == 'infer':
            pass
        elif args.operating_mode == 'debug':
            ocr_recognizer.build_model(scope="train", is_debug=True)
        
    
    