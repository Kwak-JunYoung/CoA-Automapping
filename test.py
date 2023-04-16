import torch
import pandas as pd
from models.cad4da import BERTClassifier
from data_loader import BERTDataset
from kobert.pytorch_kobert import get_pytorch_kobert_model
from kobert.utils import get_tokenizer
import gluonnlp as nlp
import numpy as np
import openpyxl

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# device = torch.device( cuda if torch.cuda.is_available() else cpu )
device = torch.device("cuda:0")

bertmodel, vocab = get_pytorch_kobert_model()

tokenizer = get_tokenizer()
tok = nlp.data.BERTSPTokenizer(tokenizer, vocab, lower=False)
tokenizer = get_tokenizer()
tok = nlp.data.BERTSPTokenizer(tokenizer, vocab, lower=False)

bertmodel, vocab = get_pytorch_kobert_model()
bertmodel, vocab = get_pytorch_kobert_model()


model_company_admin = BERTClassifier(bertmodel,  dr_rate=0.5, num_classes=6674).to(device)
model_admin_dis = BERTClassifier(bertmodel,  dr_rate=0.5, num_classes=375).to(device)

model_company_admin.load_state_dict(torch.load("./train_results/cad4da_company_admin.pt", map_location=device))
model_admin_dis.load_state_dict(torch.load("./train_results/cad4da_plain_admin_dis.pt", map_location=device))


dist_dict_df = pd.read_excel("./data/{}/dist_dict.xlsx".format("SamilCoA2023"), sheet_name='Sheet1')
admin_dict_df = pd.read_excel("./data/{}/admin_dict.xlsx".format("SamilCoA2023"), sheet_name='Sheet1')
#토큰화
tokenizer = get_tokenizer()
tok = nlp.data.BERTSPTokenizer(tokenizer, vocab, lower=False)

def predict(predict_sentence):

    data = [predict_sentence, '0']
    dataset_another = [data]

    batch_size = 64
    max_len = 40

    another_test = BERTDataset(dataset_another, 0, 1, tok, max_len, True, False)
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
        answer = admin_dict_df['관리계정'][test_eval[0]]
        return answer
        # print(test_eval[0])

def predict2(predict_sentence):

    data = [predict_sentence, '0']
    dataset_another = [data]

    batch_size = 64
    max_len = 40

    another_test = BERTDataset(dataset_another, 0, 1, tok, max_len, True, False)
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

    
df = pd.read_excel('/workspace/CoA-Automapping/houghton.xlsx', sheet_name='Sheet1')

# Iterate through the rows in the DataFrame
for index, row in df.iterrows():
    # Access the value in a column
    accntCode = row['계정코드']
    compAccnt = row['회사계정']
    adminAccnt = predict(row['회사계정'])
    compAccnt = predict2(adminAccnt)
    df.loc[index, '관리계정'] = adminAccnt
    df.loc[index, '공시용계정'] = compAccnt

df.to_excel('houghtonResult.xlsx', sheet_name='Sheet1', index=False)