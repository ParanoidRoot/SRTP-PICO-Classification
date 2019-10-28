from transformers import BertTokenizer
from pathlib import Path
import torch

from box import Box
import pandas as pd
# import collections
import os
# from tqdm import tqdm, trange
import sys
# import random
# import numpy as np
# import apex
# from sklearn.model_selection import train_test_split

import datetime
import logging

# from fast_bert.modeling import BertForMultiLabelSequenceClassification
from fast_bert.data_cls import BertDataBunch, InputExample, InputFeatures
from fast_bert.data_cls import MultiLabelTextProcessor
from fast_bert.data_cls import convert_examples_to_features
from fast_bert.learner_cls import BertLearner
from fast_bert.metrics import accuracy_multilabel, accuracy_thresh, fbeta
from fast_bert.metrics import roc_auc

# 清空 cache
torch.cuda.empty_cache()

# 配置最长显示长度以及当前日期
pd.set_option('display.max_colwidth', -1)
run_start_time = datetime.datetime.today().strftime('%Y-%m-%d_%H-%M-%S')

# 提示开始工作
os.system('cls')
os.system('cls')
print('Start'.center(25, '*'))

# 配置文件的路径
train_for = 'final'
DATA_PATH = Path('./data/%s/' % train_for)
LABEL_PATH = Path('./labels/%s/' % train_for)
MODEL_PATH = Path('./models/%s/' % train_for)
LOG_PATH = Path('./logs/%s/' % train_for)
model_state_dict = None
BERT_PRETRAINED_PATH = Path('./pretrained/')
FINETUNED_PATH = None
OUTPUT_PATH = MODEL_PATH / 'output'
OUTPUT_PATH.mkdir(exist_ok=True)
sentence_labels = ['p', 'i', 'o']
fine_grained_labels = [
    'posize', 'podisease', 'iprocedure', 'iss',
    'opatient', 'otreatment', 'prdisease', 'pogender',
    'idiagnostic', 'idiagnostictest', 'poage', 'pophyconditon',
    'idisease', 'prss', 'poss', 'potreatment',
    'poprocedure', 'poclinical', 'prbehavior', 'pomedhistory'
]

# 参数构建
args = Box({
    "run_text": "PICO-Classification-By-Bert",
    "train_size": -1,
    "val_size": -1,
    "log_path": LOG_PATH,
    "full_data_dir": DATA_PATH,
    "data_dir": DATA_PATH,
    "task_name": "PICO-Classification",
    "no_cuda": False,
    "bert_model": BERT_PRETRAINED_PATH,
    "output_dir": OUTPUT_PATH,
    "max_seq_length": (15 if train_for == 'fine_grained' else 60),
    "do_train": True,
    "do_eval": True,
    "do_lower_case": True,
    "train_batch_size": 8,
    "eval_batch_size": 16,
    "learning_rate": 5e-5,
    "num_train_epochs": 30,
    "warmup_proportion": 0.0,
    "local_rank": -1,
    "seed": 42,
    "gradient_accumulation_steps": 1,
    "optimize_on_cpu": False,
    "fp16": True,
    "fp16_opt_level": "O1",
    "weight_decay": 0.0,
    "adam_epsilon": 1e-8,
    "max_grad_norm": 1.0,
    "max_steps": -1,
    "warmup_steps": 500,
    "logging_steps": 50,
    "eval_all_checkpoints": True,
    "overwrite_output_dir": True,
    "overwrite_cache": False,
    "loss_scale": 128,
    "model_name": 'bert-base-uncased',
    "model_type": 'bert',
    'multi_gpu': False
})

# 构建 logger
logfile = str(
    LOG_PATH / 'log-{}-{}.txt'.format(run_start_time, args["run_text"])
)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S',
    handlers=[
        logging.FileHandler(logfile),
        logging.StreamHandler(sys.stdout)
    ])
logger = logging.getLogger()
logger.info(args)

# 构建 tokenizer
tokenizer = BertTokenizer.from_pretrained(
    BERT_PRETRAINED_PATH, do_lower_case=args['do_lower_case']
)

# 获取 gpu 的数目
device = torch.device('cuda')
if torch.cuda.device_count() > 1:
    args.multi_gpu = True
else:
    args.multi_gpu = False

# 设定索要的标签
label_cols = (
    sentence_labels if train_for == 'sentence' else fine_grained_labels
)

# 构建 databunch
databunch = BertDataBunch(
    args['data_dir'], LABEL_PATH, tokenizer,
    train_file='train.csv', val_file='val.csv',
    text_col='text', label_col=label_cols,
    batch_size_per_gpu=args['train_batch_size'],
    max_seq_length=args['max_seq_length'],
    multi_gpu=args.multi_gpu, multi_label=True,
    model_type=args.model_type
)

# 测试一下 databunch
print(''.center(31, '*'))
print(databunch.train_dl.dataset[0][3])
print(len(databunch.labels))

# 配置分布式
# torch.distributed.init_process_group(
#     backend="nccl",
#     init_method="tcp://localhost:23459",
#     rank=0, world_size=1
# )

# 设置计算标准
metrics = []
metrics.append({'name': 'accuracy_thresh', 'function': accuracy_thresh})
metrics.append({'name': 'roc_auc', 'function': roc_auc})
metrics.append({'name': 'fbeta', 'function': fbeta})
metrics.append({
    'name': 'accuracy_multilabel',
    'function': accuracy_multilabel
})

# 加载预训练模型
learner = BertLearner.from_pretrained_model(
    databunch, args.bert_model, metrics=metrics,
    device=device, logger=logger, output_dir=args.output_dir,
    finetuned_wgts_path=FINETUNED_PATH, warmup_steps=args.warmup_steps,
    multi_gpu=args.multi_gpu, is_fp16=args.fp16,
    multi_label=True, logging_steps=0
)

# 训练
learner.fit(args.num_train_epochs, args.learning_rate, validate=True)

# 验证
learner.validate()

# 保存
learner.save_model()
