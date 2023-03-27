import os
from pandas import pd
from argparse import ArgumentParser

BASE_PATH = "./data"

def prepare_samilCoA(data_name: str):
    # only SamilCoA2023 for now
    # data_name can be admin_dis / company_admin
    data_path = os.path.join(BASE_PATH, data_name)
    headers = ["계정코드", "회사계정", "1차번역", "계정과목", "관리계정", "공시용계정", "합산계정", "분류", "구분"]
    comp_admin_headers = ["계정코드", "1차번역", "관리계정"]
    admin_dis_headers = ["계정코드", "관리계정", "공시용계정"]
    
    usecols = []

    if data_name == "company_admin":
        usecols = comp_admin_headers
    elif data_name == "admin_dis":
        usecols = admin_dis_headers
    
    df = pd.read_excel(os.path.join(data_path, "SamilCoA2023.xlsx"), usecols=usecols)

    # if data_name == "company_admin":
    #     df = df.drop_duplicates("1차번역")
    # elif data_name == "admin_dis":
    #     df = df.drop_duplicates("관리계정")
    
    if data_name == "company_admin":
        df.to_excel(os.path.join(data_path, "preprocessed_df.xlsx"), index=False)
    elif data_name == "admin_dis":
        df.to_excel(os.path.join(data_path, "preprocessed_df.xlsx"), index=False)

if __name__ == "__main__":
    parser = ArgumentParser(description="Preprocess CoA dataset")
    parser.add_argument("--data_name", type=str, default="admin_dis")
    args = parser.parse_args()

    if args.data_name == "admin_dis":
        prepareDataset(data_name="admin_dis")
    elif args.data_name == "company_admin":
        prepareDataset(data_name="company_admin")