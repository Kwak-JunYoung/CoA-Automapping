import os
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import gluonnlp as nlp
import numpy as np
from tqdm import tqdm, tqdm_notebook

import pandas as pd
from kobert.utils import get_tokenizer
from kobert.pytorch_kobert import get_pytorch_kobert_model
from transformers import AdamW
from transformers.optimization import get_cosine_schedule_with_warmup

import time
from time import localtime

from sklearn.model_selection import train_test_split
from data_loader import BERTDataset
from models.cad4da import BERTClassifier
from train import model_train
from utils.file_io import PathManager
from utils.config import ConfigNode as CN
import argparse

import yaml

device = torch.device("cuda:0")
bertmodel, vocab = get_pytorch_kobert_model()

tokenizer = get_tokenizer()
tok = nlp.data.BERTSPTokenizer(tokenizer, vocab, lower=False)

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# 정확도 측정을 위한 함수 정의
def calc_accuracy(X, Y):
    max_vals, max_indices = torch.max(X, 1)
    train_acc = (max_indices == Y).sum().data.cpu().numpy() / \
        max_indices.size()[0]
    return train_acc


def main(config):
    tm = localtime(time.time())
    params_str = f'{tm.tm_mon}{tm.tm_mday}{tm.tm_hour}{tm.tm_min}{tm.tm_sec}'

    model_name = config.model_name
    data_name = config.data_name
    preprocess_type = config.preprocess_type
    will_test = config.test
    print(f'preprocess_type: {preprocess_type}')
    data_path = './data/{}/{}_df.xlsx'.format(data_name, preprocess_type)

    train_config = config.train_config

    batch_size = train_config.batch_size
    learning_rate = train_config.learning_rate
    max_len = train_config.max_len
    warmup_ratio = train_config.warmup_ratio
    log_interval = train_config.log_interval
    num_epochs = train_config.num_epochs
    max_grad_norm = train_config.max_grad_norm
    loss_fn = nn.CrossEntropyLoss()
    preprocess_type = config.preprocess_type

    # 전처리된 데이터. 종류는 4가지로 예상.
    df = pd.read_excel(data_path, sheet_name='Sheet1')

    # train & test 데이터로 나누기
    data_list = []

    company_admin_headers = ["계정코드", "1차번역", "관리계정"]

    comp_admin_dis_headers = ["comparative_pos", "관리계정", "공시용계정", "회사명"]
    abs_admin_dis_headers = ["index", "관리계정", "공시용계정", "회사명"]
    plain_admin_dis_headers = ["계정코드", "관리계정", "공시용계정", "회사명"]
    part_admin_dis_headers = ["계정코드", "관리계정", "공시용계정", "회사명"]

    # Confirm string concatenation of that from two columns
    if preprocess_type == "abs_admin_dis": ##
        q_list = df['index'].astype(str) + " " + df['관리계정']
        l_list = ['공시용계정']
    elif preprocess_type == "comp_admin_dis":
        q_list = df['comparative_pos'].astype(str) + " " + df['관리계정']
        l_list = df['공시용계정']
    elif preprocess_type == "company_admin":
        q_list = df['계정코드'].astype(str) + " " + df['1차번역']
        l_list = df['관리계정']
    elif preprocess_type == "part_admin_dis": ##
        # Confirm slicing being applied to all the cells in a row
        q_list = df['계정코드'].astype(str) + " " + df['관리계정']
        l_list = df['공시용계정']
    elif preprocess_type == "plain_admin_dis":
        q_list = df['계정코드'].astype(str) + " " + df['관리계정']
        l_list = df['공시용계정']
    elif preprocess_type == "admin_dis":
        q_list = df['관리계정']
        l_list = df['공시용계정']
    """
    To-do

    """

    for q, label in zip(q_list, l_list):
        data = []
        data.append(q)
        data.append(str(label))

        data_list.append(data)

    dataset_train, dataset_test = train_test_split(
        data_list, test_size=0.25, random_state=0)

    data_train = BERTDataset(dataset_train, 0, 1, tok, max_len, True, False)
    data_test = BERTDataset(dataset_test, 0, 1, tok, max_len, True, False)

    train_dataloader = torch.utils.data.DataLoader(
        data_train, batch_size=batch_size, num_workers=2)
    test_dataloader = torch.utils.data.DataLoader(
        data_test, batch_size=batch_size, num_workers=2)

    # BERT 모델 불러오기
    model = BERTClassifier(bertmodel,  dr_rate=0.5).to(device)

    # optimizer와 schedule 설정
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(
            nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in model.named_parameters() if any(
            nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate)

    t_total = len(train_dataloader) * num_epochs
    warmup_step = int(t_total * warmup_ratio)

    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_step, num_training_steps=t_total)

    trained_model = model_train(model, config, train_dataloader,
                                test_dataloader, scheduler, device, loss_fn, optimizer)

    return trained_model
    # trained_model.save_pretrained(f"./{model_name}_{data_name}_model")

def predict(predict_sentence, model, config, dist_dict_df):

    train_config = config.train_config
    batch_size = train_config.batch_size
    max_len = train_config.max_len

    data = [predict_sentence, '0']
    dataset_another = [data]

    another_test = BERTDataset(dataset_another, 0, 1, tok, max_len, True, False)
    test_dataloader = torch.utils.data.DataLoader(another_test, batch_size=batch_size, num_workers=5)
    
    model.eval()

    for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(test_dataloader):
        token_ids = token_ids.long().to(device)
        segment_ids = segment_ids.long().to(device)

        valid_length= valid_length
        label = label.long().to(device)

        out = model(token_ids, valid_length, segment_ids)


        test_eval=[]
        for i in out:
            logits=i
            logits = logits.detach().cpu().numpy()
            test_eval.append(np.argmax(logits))
            
        print(dist_dict_df['공시용계정'][test_eval[0] - 1])
        # Output needs to be settled. Additional coding needs to be done in preprocess_data by creating an excel sheet that has info about dis_enums.
        # print(test_eval[0])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_name",
        type=str,
        default="cad4da",
        help="The name of the model to train. \
            The possible models are in [akt, cl4kt]. \
            The default model is cl4kt.",
    )
    parser.add_argument(
        "--data_name",
        type=str,
        default="SamilCoA2023",
        help="The name of the dataset to use in training.",
    )
    parser.add_argument(
        "--max_len", type=float, default=64, help="max length"
    )
    parser.add_argument(
        "--batch_size", type=float, default=64, help="batch size"
    )
    parser.add_argument(
        "--warmup_ratio", type=float, default=0.1, help="warmup ratio"
    )
    parser.add_argument(
        "--num_epochs", type=int, default=10, help="number of epochs"
    )
    parser.add_argument(
        "--max_grad_norm", type=float, default=1, help="max grad norm"
    )
    parser.add_argument(
        "--log_interval", type=float, default=200, help="log interval"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=5e-5, help="learning rate"
    )
    parser.add_argument(
        "--hidden_size", type=float, default=768, help="hidden size"
    )
    parser.add_argument(
        "--num_classes", type=float, default=375, help="Number of categories to be classified"
    )
    parser.add_argument(
        "--preprocess_type", type = str, default="admin_dis", help="preprocess type"
    )
    parser.add_argument(
        "--will_test", type=str, default="false", help="will the model be tested manually"
    )

    base_cfg_file = PathManager.open("configs/example.yaml", "r")
    base_cfg = yaml.safe_load(base_cfg_file)

    args = parser.parse_args()

    cfg = CN(base_cfg)
    cfg.set_new_allowed(True)
    cfg.model_name = args.model_name
    cfg.data_name = args.data_name
    cfg.preprocess_type = args.preprocess_type
    cfg.will_test = args.will_test

    cfg.train_config.batch_size = int(args.batch_size)
    cfg.train_config.learning_rate = args.learning_rate
    cfg.train_config.max_len = args.max_len
    cfg.train_config.warmup_ratio = args.warmup_ratio
    cfg.train_config.num_epochs = args.num_epochs
    cfg.train_config.max_grad_norm = args.max_grad_norm
    cfg.train_config.log_interval = args.log_interval

    if args.model_name == 'cad4da':
        cfg.cad4da_config.hidden_size = args.hidden_size
        cfg.cad4da_config.num_classes = args.num_classes

    cfg.freeze()

    trained_model = main(cfg)

    if(cfg.will_test == "true"):
        #질문 무한반복하기! 0 입력시 종료
        dist_dict_df = pd.read_excel("./data/{}/dist_dict.xlsx".format(cfg.data_name), sheet_name='Sheet1')
        end = 1
        while end == 1 :
            sentence = input("하고싶은 말을 입력해주세요 : ")
            if sentence == 0 :
                break
            predict(sentence, trained_model, cfg, dist_dict_df)
            print("\n")
