import argparse
import logging
import os
import re
import random
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import sys
import torch
import numpy as np
from collections import OrderedDict
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUC_ORDER = {y: x for x, y in enumerate(["A", "T", "C", "G", "N"])}
NUCLEOTIDES = sorted([x for x in NUC_ORDER.keys()])
NUC_COUNT = len(NUC_ORDER)
seed = 6
os.environ['PYTHONHASHSEED'] = str(seed)
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True