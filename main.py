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

# from kobert.utils import get_tokenizer
# from kobert_transformers import get_tokenizer
# from kobert.pytorch_kobert import get_pytorch_kobert_model

# Recent
from kobert_tokenizer import KoBERTTokenizer
from transformers import BertModel

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
# from test import predict

device = torch.device("cuda:0")
# bertmodel, vocab = get_pytorch_kobert_model()
# tokenizer = get_tokenizer()

tokenizer = KoBERTTokenizer.from_pretrained('skt/kobert-base-v1')
bertmodel = BertModel.from_pretrained('skt/kobert-base-v1', return_dict=False)
vocab = nlp.vocab.BERTVocab.from_sentencepiece(tokenizer.vocab_file, padding_token='[PAD]')

tok = tokenizer.tokenize
# tok = tokenizer
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
    will_test = config.will_test
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

    if model_name == "cad4da":
        cad4da_config = config.cad4da_config
        num_classes = cad4da_config.num_classes
        hidden_size = cad4da_config.hidden_size

    # 전처리된 데이터. 종류는 4가지로 예상.
    df = pd.read_excel(data_path, sheet_name='Sheet1')
    adminDf = pd.read_excel('./data/SamilCoA2023/admin_dict.xlsx', sheet_name='Sheet1')
    distDf = pd.read_excel('./data/SamilCoA2023/dist_dict.xlsx', sheet_name='Sheet1')
    # train & test 데이터로 나누기
    data_list = []

    company_admin_headers = ["계정코드", "1차번역", "관리계정"]

    comp_admin_dis_headers = ["comparative_pos", "관리계정", "공시용계정", "회사명"]
    abs_admin_dis_headers = ["index", "관리계정", "공시용계정", "회사명"]
    plain_admin_dis_headers = ["계정코드", "관리계정", "공시용계정", "회사명"]
    part_admin_dis_headers = ["계정코드", "관리계정", "공시용계정", "회사명"]

    idx_admin_dict = adminDf.to_dict()['관리계정']
    admin_idx_dict = {y:x for x, y in idx_admin_dict.items()}
    idx_dist_dict = distDf.to_dict()['공시용계정']
    dist_idx_dict = {y:x for x, y in idx_dist_dict.items()}  

    # Confirm string concatenation of that from two columns
    if preprocess_type == "abs_admin_dis": ##
        q_list = df['index'].astype(str) + " " + df['관리계정']
        l_list = df['공시용계정']
    elif preprocess_type == "comp_admin_dis":
        q_list = df['comparative_pos'].astype(str) + " " + df['관리계정']
        l_list = df['공시용계정']
    elif preprocess_type == "part_admin_dis": ##
        # Confirm slicing being applied to all the cells in a row
        q_list = df['계정코드'].astype(str).str[:2] + " " + df['관리계정']
        l_list = df['공시용계정']
    elif preprocess_type == "plain_admin_dis":
        q_list = df['계정코드'].astype(str) + " " + df['관리계정']
        l_list = df['공시용계정']

    elif preprocess_type == "abs_company_admin": ##
        q_list = df['index'].astype(str) + " " + df['1차번역']
        l_list = df['관리계정']
    elif preprocess_type == "comp_company_admin":
        q_list = df['comparative_pos'].astype(str) + " " + df['1차번역']
        l_list = df['관리계정']
    elif preprocess_type == "part_company_admin": ##
        # Confirm slicing being applied to all the cells in a row
        q_list = df['계정코드'].astype(str).str[:2] + " " + df['1차번역']
        l_list = df['관리계정']
    elif preprocess_type == "plain_company_admin":
        q_list = df['계정코드'].astype(str) + " " + df['1차번역'].astype(str)
        l_list = df['관리계정']

    elif preprocess_type == "company_admin":
        q_list = df['1차번역']
        l_list = df['관리계정']
    elif preprocess_type == "admin_dis":
        q_list = df['관리계정']
        l_list = df['공시용계정']        

    """
    To-do
    Filling data_list under the condition of preprocess type.
    """

    if "company_admin" in preprocess_type:
        for q, label in zip(q_list, l_list):
            data = []
            data.append(str(q))
            data.append(str(admin_idx_dict[label]))

            data_list.append(data)

    elif "admin_dis" in preprocess_type:
        for q, label in zip(q_list, l_list):
            data = []
            data.append(str(q))
            data.append(str(dist_idx_dict[label]))

            data_list.append(data)

    dataset_train, dataset_test = train_test_split(
        data_list, test_size=0.25, random_state=0)
    # def __init__(self, dataset, sent_idx, label_idx, bert_tokenizer, vocab, max_len, pad, pair):
    data_train = BERTDataset(dataset=dataset_train, sent_idx=0, label_idx=1, bert_tokenizer=tok, vocab=vocab, max_len=max_len, pad=True, pair=False)
    data_test = BERTDataset(dataset=dataset_test, sent_idx=0, label_idx=1, bert_tokenizer=tok, vocab=vocab, max_len=max_len, pad=True, pair=False)

    train_dataloader = DataLoader(
        data_train, batch_size=batch_size, num_workers=2)
    test_dataloader = DataLoader(
        data_test, batch_size=batch_size, num_workers=2)

    # BERT 모델 불러오기
    # 여기에 hyperparameter: num_classes 추가
    model = BERTClassifier(bertmodel,  dr_rate=0.5, num_classes=num_classes).to(device)

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
        "--num_classes", type=int, default=375, help="Number of categories to be classified"
    )
    parser.add_argument(
        "--preprocess_type", type = str, default="admin_dis", help="preprocess type"
    )
    parser.add_argument(
        "--will_test", type=str, default="false", help="will the model be tested manually"
    )
    parser.add_argument(
        "--will_save", type=str, default="true", help="Decide whether to save the trained model"
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
    cfg.will_save = args.will_save

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

    if(cfg.will_save == "true"):
        torch.save(trained_model.state_dict(), "./data/{}/{}_{}_model.pt".format(cfg.data_name, cfg.model_name, cfg.preprocess_type))
        # pickle.dump(trained_model, open("./data/{}/{}_{}_config.pkl".format(cfg.data_name, cfg.model_name, cfg.data_name), "wb"))

    # if(cfg.will_test == "true"):
    #     #질문 무한반복하기! 0 입력시 종료
    #     dist_dict_df = pd.read_excel("./data/{}/dist_dict.xlsx".format(cfg.data_name), sheet_name='Sheet1')
    #     end = 1
    #     while end == 1 :
    #         sentence = input("하고싶은 말을 입력해주세요 : ")
    #         if sentence == 0 :
    #             break
    #         predict(sentence, trained_model, cfg, dist_dict_df)
    #         print("\n")
    

