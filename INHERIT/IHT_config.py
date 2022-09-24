# Author: Zeheng Bai
##### INHERIT DNABERT MODELS #####
from basicsetting import *

### You should replace '.' with their paths if they do not in $Your_current_workpath/INHERIT/INHERIT ###
### Please put the pre-trained models in the same path of this code. If not, please use their absolute paths ###
CONFIG_PATH = './transformers/dnabert-config/bert-config-6'
BAC_TR_PATH = 'bac_training.txt'
PHA_TR_PATH = 'pha_training.txt'
BAC_VAL_PATH = 'bac_val.txt'
PHA_VAL_PATH = 'pha_val.txt'
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
BAC_PTRMODEL = 'bac_pretrained_model'
PHA_PTRMODEL = 'pha_pretrained_model'
