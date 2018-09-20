import argparse
import json
import os, shutil
import os.path as osp

def parse_args():
    desc = "TensorFlow Implementation of CRNN models"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--version', type=int, default=1,
                        help='version for this model')
    parser.add_argument('--model_name', type=str, default='UNet',
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
    
    args.CKPT_DIR = "ckpt/{}/{}".format(SAVE_DIR_NAME, args.model_name_with_version)
    args.LOG_DIR = "logs/{}/{}".format(SAVE_DIR_NAME, args.model_name_with_version)
    if not args.reload:
        if osp.exists(args.CKPT_DIR):
            shutil.rmtree(args.CKPT_DIR, ignore_errors=True)
        else:
            os.makedirs(args.CKPT_DIR)
        
        if osp.exists(args.LOG_DIR):
            shutil.rmtree(args.LOG_DIR, ignore_errors=True)
        else:
            os.makedirs(args.LOG_DIR)
            
    args.DEBUG_DIR = "debug"
    
    return args

def args2json(args):
    d = vars(args)
    with open(osp.join(args.CKPT_DIR, "meta_infomation.json"), 'w+') as fp:
        json.dump(d, fp=fp, indent=2)
    return args

