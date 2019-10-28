from transformers import BertTokenizer
from pathlib import Path
import torch

from box import Box
import pandas as pd
import os
import sys
import datetime
import logging

from fast_bert.data_cls import BertDataBunch, InputExample, InputFeatures
from fast_bert.data_cls import MultiLabelTextProcessor
from fast_bert.data_cls import convert_examples_to_features
from fast_bert.learner_cls import BertLearner
from fast_bert.metrics import accuracy_multilabel, accuracy_thresh, fbeta
from fast_bert.metrics import roc_auc
from fast_bert.prediction import BertClassificationPredictor

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
ANS_PATH = Path('./ans/%s/' % train_for)
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
    "num_train_epochs": 20,
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

# 开始构建预测模型
predictor = BertClassificationPredictor(
    model_path=args.output_dir / 'model_out',
    label_path=LABEL_PATH,
    multi_label=True,
    model_type=args.model_type,
    do_lower_case=True
)

# 获取测试数据
output = predictor.predict_batch(
    list(
        pd.read_csv(
            str(DATA_PATH.joinpath('test.csv').absolute())
        )['text'].values
    )
)

# 将预测结果输出
pd.DataFrame(output).to_csv(
    str(DATA_PATH.joinpath('output_bert.csv').absolute())
)

# 预测结果读入
results = pd.read_csv(
    str(DATA_PATH.joinpath('output_bert.csv').absolute())
)

# 预测结果构成一个 pd 对象
preds = pd.DataFrame([{item[0]: item[1] for item in pred} for pred in output])
print(preds.head())

test_df = pd.read_csv(
    str(DATA_PATH.joinpath('test.csv').absolute())
)
print(test_df.head())
output_df = pd.merge(
    test_df, preds, how='left',
    left_index=True, right_index=True
)

output_df.to_csv(
    str(
        ANS_PATH.joinpath('ans.csv').absolute()
    )
)

print(''.center(31, '*'))
final_pd = pd.read_csv(
    str(
        ANS_PATH.joinpath('ans.csv').absolute()
    )
)
print(final_pd.head())
