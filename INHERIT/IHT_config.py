# Author: Zeheng Bai
##### INHERIT DNABERT MODELS #####
from basicsetting import *

### You should replace '.' with their paths if they do not in $Your_current_workpath/INHERIT/INHERIT ###
CONFIG_PATH = './transformers/dnabert-config/bert-config-6'
#BAC_TR_PATH = './bac_training.txt'
#PHA_TR_PATH = './pha_training.txt'
#BAC_VAL_PATH = './bac_val.txt'
#PHA_VAL_PATH = './pha_val.txt'
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
BAC_PTRMODEL = 'bac_pretrained_model'
PHA_PTRMODEL = 'pha_pretrained_model'
