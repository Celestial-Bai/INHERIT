# Author: Zeheng Bai
##### IGNITER DNABERT MODELS #####
from basicsetting import *

CONFIG_PATH = '/yshare2/ZETTAI_path_WA_slash_home_KARA/home/zeheng/DNABERT_embedding/examples/transformers/dnabert-config/bert-config-6'
BAC_TR_PATH = '/yshare2/ZETTAI_path_WA_slash_home_KARA/home/zeheng/seeker/train_model/bac_training_3.txt'
PHA_TR_PATH = '/yshare2/ZETTAI_path_WA_slash_home_KARA/home/zeheng/seeker/train_model/pha_training_3.txt'
BAC_VAL_PATH = '/yshare2/ZETTAI_path_WA_slash_home_KARA/home/zeheng/seeker/train_model/bac_val_3.txt'
PHA_VAL_PATH = '/yshare2/ZETTAI_path_WA_slash_home_KARA/home/zeheng/seeker/train_model/pha_val_3.txt'
PATIENCE = 3
KMERS = 6
SEGMENT_LENGTH = 500
TR_BATCHSIZE = 64
VAL_BATCHSIZE = 32
TR_WORKERS = 6
VAL_WORKERS = 6
EPOCHS = 100
THRESHOLD = 0.5
LEARNING_RATE = 1e-5
BAC_PTRMODEL = '/yshare2/ZETTAI_path_WA_slash_home_KARA/home/zeheng/DNABERT_embedding/examples/bac2-1000'
PHA_PTRMODEL = '/yshare2/ZETTAI_path_WA_slash_home_KARA/home/zeheng/DNABERT_embedding/examples/pha3-5000'
