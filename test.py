import torch
import pandas as pd
from models.cad4da import BERTClassifier
from data_loader import BERTDataset
# from kobert.pytorch_kobert import get_pytorch_kobert_model
# from kobert.utils import get_tokenizer
import gluonnlp as nlp
import numpy as np
import openpyxl

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# device = torch.device( cuda if torch.cuda.is_available() else cpu )
device = torch.device("cuda:0")


from transformers import BertModel
from kobert_tokenizer import KoBERTTokenizer
tokenizer = KoBERTTokenizer.from_pretrained('skt/kobert-base-v1')
bertmodel = BertModel.from_pretrained('skt/kobert-base-v1', return_dict=False)
vocab = nlp.vocab.BERTVocab.from_sentencepiece(tokenizer.vocab_file, padding_token='[PAD]')

# bertmodel, vocab = get_pytorch_kobert_model()

# tokenizer = get_tokenizer()
# tok = nlp.data.BERTSPTokenizer(tokenizer, vocab, lower=False)

# bertmodel, vocab = get_pytorch_kobert_model()

# parameter로 받아들여보기
model_company_admin = BERTClassifier(bertmodel,  dr_rate=0.5, num_classes=4950).to(device)
model_admin_dis = BERTClassifier(bertmodel,  dr_rate=0.5, num_classes=375).to(device)

model_company_admin.load_state_dict(torch.load("./train_results/cad4da_plain_company_admin_model.pt", map_location=device))
model_admin_dis.load_state_dict(torch.load("./train_results/cad4da_plain_admin_dis_model.pt", map_location=device))

dataset_name = "SamilCoA2023"

dist_dict_df = pd.read_excel("./data/{}/dist_dict.xlsx".format(dataset_name), sheet_name='Sheet1')
admin_dict_df = pd.read_excel("./data/{}/admin_dict.xlsx".format(dataset_name), sheet_name='Sheet1')
ghbg_df = pd.read_excel("./data/{}/GHBG.xlsx".format(dataset_name), sheet_name='Sheet1')

#토큰화

tok = tokenizer.tokenize

def predict(predict_sentence):
    print("회->관")
    data = [predict_sentence, '0']
    dataset_another = [data]

    batch_size = 64
    max_len = 40
    # data_test = BERTDataset(dataset=dataset_test, sent_idx=0, label_idx=1, bert_tokenizer=tok, vocab=vocab, max_len=max_len, pad=True, pair=False)

    another_test = BERTDataset(dataset_another, 0, 1, tok, vocab, max_len, True, False)
    test_dataloader = torch.utils.data.DataLoader(another_test, batch_size=batch_size, num_workers=5)
    
    model_company_admin.eval()

    for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(test_dataloader):
        token_ids = token_ids.long().to(device)
        segment_ids = segment_ids.long().to(device)

        valid_length= valid_length
        label = label.long().to(device)

        out = model_company_admin(token_ids, valid_length, segment_ids)

        test_eval=[]
        for i in out:
            logits=i
            logits = logits.detach().cpu().numpy()
            test_eval.append(np.argmax(logits))

        # answer = dist_dict_df['공시용계정'][test_eval[0]]
        print("{} -> {}".format(predict_sentence, test_eval))
        answer = admin_dict_df['관리계정'][test_eval[0]]
        return answer
        # print(test_eval[0])

def predict2(predict_sentence):

    data = [predict_sentence, '0']
    dataset_another = [data]

    batch_size = 64
    max_len = 40

    another_test = BERTDataset(dataset_another, 0, 1, tok, vocab, max_len, True, False)
    test_dataloader = torch.utils.data.DataLoader(another_test, batch_size=batch_size, num_workers=5)
    
    model_admin_dis.eval()

    for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(test_dataloader):
        token_ids = token_ids.long().to(device)
        segment_ids = segment_ids.long().to(device)

        valid_length= valid_length
        label = label.long().to(device)

        out = model_admin_dis(token_ids, valid_length, segment_ids)

        test_eval=[]
        for i in out:
            logits=i
            logits = logits.detach().cpu().numpy()
            test_eval.append(np.argmax(logits))

        answer = dist_dict_df['공시용계정'][test_eval[0]]
        # answer = admin_dict_df['관리계정'][test_eval[0]]
        return answer
        # print(test_eval[0])

# 합산, 분류, 구분 불러오기
def predict3(compAccnt):
    for index, row in ghbg_df.iterrows():
        if row[0] == compAccnt:
            return row


df = pd.read_excel('./target/houghton.xlsx', sheet_name='Sheet1')

# Iterate through the rows in the DataFrame
for index, row in df.iterrows():
    # Access the value in a column
    # Casting modified
    accntCode = str(row['계정코드'])                            # 계정코드
    compAccnt = row['1차번역']                                 # 회사계정
    adminAccnt = predict(accntCode + " " + compAccnt)           # 관리계정
    discAccnt = predict2(accntCode + " " + adminAccnt)          # 공시용계정 
    ghbgAccnt = predict3(discAccnt)                             # 합산계정, 분류, 구분

    df.loc[index, '관리계정'] = adminAccnt
    df.loc[index, '공시용계정'] = discAccnt
    df.loc[index, '합산계정'] = ghbgAccnt[0]
    df.loc[index, '분류'] = ghbgAccnt[1]
    df.loc[index, '구분'] = ghbgAccnt[2]

    # print("계정코드: {}\t회사계정: {}\t관리계정: {}\t공시용계정: {}\t합산계정: {}\t분류: {}\t구분: {}".format(accntCode, compAccnt, adminAccnt, discAccnt, ghbgAccnt[0], ghbgAccnt[1], ghbgAccnt[2]))

    # 합산계정, 분류, 구분

df.to_excel('./result/houghtonResult.xlsx', sheet_name='Sheet1', index=False)