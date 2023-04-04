# import torch
# import pandas as pd
# from models.cad4da import BERTClassifier
# from data_loader import BERTDataset
# from kobert.pytorch_kobert import get_pytorch_kobert_model
# from kobert.utils import get_tokenizer
# import gluonnlp as nlp
# import numpy as np
# from flask import Flask, request, jsonify

# device = torch.device('cpu')
# bertmodel, vocab = get_pytorch_kobert_model()

# tokenizer = get_tokenizer()
# tok = nlp.data.BERTSPTokenizer(tokenizer, vocab, lower=False)

# app = Flask(__name__)

# bertmodel, vocab = get_pytorch_kobert_model()

# device = torch.device('cpu')

# model = BERTClassifier(bertmodel,  dr_rate=0.5).to(device)
# model.load_state_dict(torch.load("./data/SamilCoA2023/cad4da_SamilCoA2023_model.pt", map_location=device))

# dist_dict_df = pd.read_excel("./data/{}/dist_dict.xlsx".format("SamilCoA2023"), sheet_name='Sheet1')

# app.route('/ad_dis', methods=['POST'])
# def predict():

#     predict_sentence = request.form['myName']

#     batch_size = 64
#     max_len = 40

#     data = [predict_sentence, '0']
#     dataset_another = [data]

#     another_test = BERTDataset(dataset_another, 0, 1, tok, max_len, True, False)
#     test_dataloader = torch.utils.data.DataLoader(another_test, batch_size=batch_size, num_workers=2)
    
#     model.eval()

#     for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(test_dataloader):
#         token_ids = token_ids.long().to(device)
#         segment_ids = segment_ids.long().to(device)

#         valid_length= valid_length
#         label = label.long().to(device)

#         out = model(token_ids, valid_length, segment_ids)

#         test_eval=[]
#         for i in out:
#             logits=i
#             logits = logits.detach().cpu().numpy()
#             test_eval.append(np.argmax(logits))
            
#         return jsonify({'dist': dist_dict_df['공시용계정'][test_eval[0]]})
#         # Output needs to be settled. Additional coding needs to be done in preprocess_data by creating an excel sheet that has info about dis_enums.
#         # curl -X POST http://localhost:5000/predict -F "file=@kitten.jpg"
#         # print(test_eval[0])