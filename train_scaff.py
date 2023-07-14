# %%
import torch
import numpy as np
import json
import datetime
import deepchem as dc
from dataloader import MolDataset
from model import BertClassifier
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModel, TrainingArguments, Trainer, IntervalStrategy
from splitter import random_scaffold_split

# %%
from config import *
cfg = update_cfg(cfg)

def compute_metrics(p):
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
    labels = p.label_ids
    pred = p.predictions.argmax(-1)
    auc = roc_auc_score(y_true=labels, y_score=pred)

    return {'auc': auc}

import logging
def get_logger(log_dir='./trace.log'):
    log = logging.getLogger()
    console = logging.StreamHandler()
    log.addHandler(console)
    formatter = logging.Formatter(
        fmt='[%(levelname)s] %(asctime)s [%(filename)12s:%(lineno)5d]:\t%(message)s'
    )
    console.setFormatter(formatter)
    log.setLevel(logging.DEBUG)
    log.propagate = False

    fh = logging.FileHandler(filename=log_dir, encoding='utf-8', mode='a')
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    log.addHandler(fh)

    return log


def set_seed(seed):
    # random.seed(seed)
    # os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)
    # torch.backends.cudnn.deterministic = True

seed = 0
set_seed(seed)

# %%
# -------------- params -----------------

# dataset_name = 'bace'
dataset_name = cfg.dataset

lm_model_name = cfg.lm.model.name
# lm_model_name = 'microsoft/deberta-base'
# lm_model_name = 'mossaic-candle/adaptive-lm-molecules'
# lm_model_name = 'roberta-base'
# lm_model_name = 'bert-base-cased'
# lm_model_name = 'bert-base-uncased'
# lm_model_name = 'prajjwal1/bert-tiny'
lm_model_feat_shrink = ''

use_gpt = cfg.lm.use_gpt


dataset_path = 'dataset/gpt/{}_gpt3_result.json'.format(dataset_name)

# model_name = cfg.lm.model.name
feat_shrink = cfg.lm.model.feat_shrink

weight_decay = cfg.lm.train.weight_decay
dropout = cfg.lm.train.dropout
att_dropout = cfg.lm.train.att_dropout
cla_dropout = cfg.lm.train.cla_dropout
batch_size = cfg.lm.train.batch_size
epochs = cfg.lm.train.epochs
warmup_epochs = cfg.lm.train.warmup_epochs
eval_patience = cfg.lm.train.eval_patience
grad_acc_steps = cfg.lm.train.grad_acc_steps
lr = cfg.lm.train.lr

semi_ratio = cfg.semi_ratio
split_traintest_seed = 0
split_trainval_seed = 0

# dataset
texts, labels = [], []
all_smiles = []
with open(dataset_path, 'r', encoding='UTF-8') as f:
    all_infos = json.load(f)
for line in all_infos:
    # texts.append(line['data'])  # smiles
    query = line['question']
    # location_tag = 'briefly:'
    location_tag = 'Here is the SMILES string: '
    smiles_location = query.find(location_tag) + len(location_tag)
    smiles = query[smiles_location:]
    all_smiles.append(smiles)
    if not use_gpt:
        texts.append(smiles)
    else:
        texts.append(line['answer'])  # ChatGPT answer
    
    labels.append(int(line['label']))

# print(len(labels), labels)

tokenizer = AutoTokenizer.from_pretrained(lm_model_name)
X = tokenizer(texts, padding=True, truncation=True, max_length=512)

dataset = MolDataset(dataset_path=dataset_path, encodings=X)

# ---------------- logger ------------------
time_prefix = datetime.datetime.strftime(datetime.datetime.now(), '%m-%d-%H-%M')
exp_tag = '{}_epochs={}_semiratio={}'.format(time_prefix, epochs, semi_ratio)
log_dir = 'log_scaff/{}/{}/'.format(dataset_name, lm_model_name.replace('/', '-'))
os.makedirs(log_dir, exist_ok=True)

logger = get_logger(log_dir=log_dir+exp_tag+'.log')


# ---------------- 5-fold scaffold split ------------------
for scaff_seed in [0, 1, 2, 3, 4]:

    dc_dataset = dc.data.DiskDataset.from_numpy(X=all_smiles, y=labels, ids=all_smiles)
    idx_subtrain, idx_val, idx_test = random_scaffold_split(dc_dataset, smiles_list=all_smiles, seed=scaff_seed)

    y_train = [labels[idx] for idx in idx_subtrain]

    n_labels = torch.unique(torch.tensor(y_train)).shape[0]
    n_dataset = len(labels)

    train_dataset = torch.utils.data.Subset(dataset, idx_subtrain)
    val_dataset = torch.utils.data.Subset(dataset, idx_val)
    test_dataset = torch.utils.data.Subset(dataset, idx_test)

    # print(len(train_dataset))

    # Define pretrained tokenizer and model
    bert_model = AutoModel.from_pretrained(lm_model_name)
    cls_model = BertClassifier(bert_model, n_labels=n_labels, feat_shrink=lm_model_feat_shrink)

    cls_model.config.dropout = dropout
    cls_model.config.attention_dropout = att_dropout

    trainable_params = sum(p.numel() for p in cls_model.parameters() if p.requires_grad)
    print('Number of parameters: {}M'.format(trainable_params/1e6))

    # ------------- train -------------------
    # Define training parameters
    eq_batch_size = batch_size
    train_steps = n_dataset // eq_batch_size + 1
    # eval_steps = n_dataset // eq_batch_size
    eval_steps = 100000
    warmup_steps = int(warmup_epochs * train_steps)

    print('eval_steps: {}, warmup_steps: {}', eval_steps, warmup_steps)

    # Define Trainer
    args = TrainingArguments(
        output_dir='/data00/qc/smiles',
        # do_train=True,
        # do_eval=True,
        eval_steps=eval_steps,
        evaluation_strategy=IntervalStrategy.STEPS,
        # evaluation_strategy="epoch",
        save_steps=eval_steps,
        learning_rate=lr,
        weight_decay=weight_decay,
        # save_total_limit=1,
        load_best_model_at_end=True,
        gradient_accumulation_steps=grad_acc_steps,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size*8,
        warmup_steps=warmup_steps,
        num_train_epochs=epochs,
        dataloader_num_workers=1,
        fp16=True,
        # dataloader_drop_last=True,
        metric_for_best_model='auc',
        # label_names=["labels"],
    )
    trainer = Trainer(
        model=cls_model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        # callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    # Train pre-trained model
    trainer.train()

    trainer.evaluate()

    logger.info('-'*80)
    predictions = trainer.predict(test_dataset)
    logger.info('predictions: %s', predictions)



