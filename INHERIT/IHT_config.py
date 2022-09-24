# Author: Zeheng Bai
##### INHERIT DNABERT MODELS #####
from basicsetting import *

### You should replace '.' with their paths if they do not in $Your_current_workpath/INHERIT/INHERIT ###
### Please put the pre-trained models in the same path of this code. If not, please use their absolute paths ###
CONFIG_PATH = './transformers/dnabert-config/bert-config-6'
BAC_TR_PATH = '/rshare1/ZETTAI_path_WA_slash_home_KARA/home/zeheng/seeker/train_model/bac_training_3.txt'
PHA_TR_PATH = '/rshare1/ZETTAI_path_WA_slash_home_KARA/home/zeheng/seeker/train_model/pha_training_3.txt'
BAC_VAL_PATH = '/rshare1/ZETTAI_path_WA_slash_home_KARA/home/zeheng/seeker/train_model/bac_val_3.txt'
PHA_VAL_PATH = '/rshare1/ZETTAI_path_WA_slash_home_KARA/home/zeheng/seeker/train_model/pha_val_3.txt'
PATIENCE = 3
KMERS = 6
SEGMENT_LENGTH = 500
TR_BATCHSIZE = 64
VAL_BATCHSIZE = 32
TR_WORKERS = 3
VAL_WORKERS = 3
EPOCHS = 100
THRESHOLD = 0.5
LEARNING_RATE = 1e-5
BAC_PTRMODEL = 'bac_pretrained_model_new'
PHA_PTRMODEL = 'pha_pretrained_model_new'
