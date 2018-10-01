import argparse
import json
import os, shutil
import os.path as osp

def parse_args():
    desc = "TensorFlow Implementation of CRNN models"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--version', type=int, default=1,
                        help='version for this model')
    parser.add_argument('--model_name', type=str, default='CRNN',
                        help='model name for using')
    parser.add_argument('--reload', action='store_true', dest='reload', default=False,
                        help='reload model from past state')
    parser.add_argument('--operating_mode', type=str, default='train',
                        help="operating mode for this model")
    parser.add_argument('--optimizer', type=str, default='RMSProp', 
                        help="optimizer name for train")
    parser.add_argument('--gpu', type=int, default=0,
                        help='gpu id for using')
    parser.add_argument('--desc', type=str, default='desc',
                        help='description for this learning')
    parser.add_argument('--lambda1', type=float, default=1.0,
                        help='Scale value for Reconstructed Loss')
    parser.add_argument('--loss', type=str, default='MSE',
                        help='Scale value for Reconstructed Loss')
    parser.add_argument('--lr_value', type=float, default=0.001,
                        help="initial learning rate value")
    parser.add_argument('--lr_method', type=str, default='unchange_decay',
                        help="fix | time_decay | step_decay | unchange_decay")
    parser.add_argument('--lr_decay', type=float, default=0.9,
                        help="learning rate decay value")
    parser.add_argument('--lr_drop_step', type=int, default=10000,
                        help="lr_value drop step: only use for lr_method->step_decay")
    parser.add_argument('--testset', type=str, default='ic13_1015,ic13_857',
                        help="benchmark testset list for scoring")

    return check_args(parser.parse_args())

def check_args(args):
    args.model_name_with_version = "{}_{}_{}_{}".format(args.model_name, args.loss, args.optimizer, args.desc)
    args.testset_names = [item.strip() for item in args.testset.split(',')]
    
    args.IMAGE_WIDTH = 256
    args.IMAGE_HEIGHT = 32
    args.IMAGE_CHANNELS = 1

    args.SEQ_LENGTH = 25
    args.BATCH_SIZE = 128
    #BATCH_SIZE = 1024

    args.MAX_UNCHANGED = 15

    args.RNN_HIDDEN_SIZE = 512
    args.RNN_STACK_SIZE = 2
    args.LOG_INTERVAL = 10
    args.VALID_INTERVAL = 100

    args.MAX_EPOCHS = 50

    args.MIN_LEARNING_RATE = 1e-7
    
    args.DECODER_TYPE = 'attention_based'
    
    args.TRAIN_MAIN_DATASET_DIR = "data_segment_2.7M"
    args.TRAIN_SUB_DATASET_DIR = "data_recognition_11.6M"
    args.TEST_DATASET_DIR = 'benchmark_db/icdar13_1015_segment_test'
    args.INFER_DATASET_DIR = 'benchmark_db/benchmark_infers'
    
    args.INFER_RESULT_DIR = 'result_infers'
    
    SAVE_DIR_NAME = "recog_11.6M_seg_2.7M"
    
    args.CKPT_DIR = "ckpt/{}".format(SAVE_DIR_NAME)
    args.LOG_DIR = "logs/{}".format(SAVE_DIR_NAME)
    if not args.reload:
        ckpt_dir = osp.join(args.CKPT_DIR, args.model_name_with_version)
        if osp.exists(ckpt_dir):
            shutil.rmtree(ckpt_dir, ignore_errors=True)
        else:
            os.makedirs(ckpt_dir)
        
        log_dir = osp.join(args.LOG_DIR, args.model_name_with_version)
        if osp.exists(log_dir):
            shutil.rmtree(log_dir, ignore_errors=True)
        else:
            os.makedirs(log_dir)
            
    args.DEBUG_DIR = "debug"
    
    return args

def args2json(args):
    d = vars(args)
    with open(osp.join(args.CKPT_DIR, args.model_name_with_version, "meta_infomation.json"), 'w+') as fp:
        json.dump(d, fp=fp, indent=2)
    return args

class LR_Generator:
    def __init__(self, type_name, initial_lr, decay_rate, args, drop_step=None):
        self.type_name = type_name # TODO: fix, time_decay, step_decay, clr
        self.decay_rate = decay_rate
        self.drop_step = drop_step
        self.initial_lr = initial_lr
        self.args = args

    def get_lr(self, current_lr, current_step=None, unchanged_count=None):
        if self.type_name == 'fix':
            return current_lr, False
        elif self.type_name == 'time_decay':
            current_lr *= (1. / (1. + self.decay_rate * current_step))
            return current_lr, False
        elif self.type_name == 'step_decay':
            current_lr = self.initial_lr * math.pow(self.decay_rate, (current_step / self.drop_step))
            return current_lr, False
        elif self.type_name == 'unchange_decay':
            if unchanged_count >= self.args.MAX_UNCHANGED:
                return current_lr * self.decay_rate, True
            else:
                return current_lr, False

