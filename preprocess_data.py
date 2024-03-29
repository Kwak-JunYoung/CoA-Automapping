import os
from argparse import ArgumentParser
import pandas as pd
import numpy as np
from IPython.display import display 

BASE_PATH = "./data"

# 공시용계정과, 그의 번호
def get_dist_info(
    originalDf
):
    dist_list = originalDf["공시용계정"].drop_duplicates(keep="first").to_list()
    dist_dict = {}
    for i in range(len(dist_list)):
        dist_dict[dist_list[i]] = i

    dist_dict_df = pd.DataFrame(data=dist_dict, index=[0])
    dist_dict_df = (dist_dict_df.T)
    dist_dict_df.to_excel('dist_dict.xlsx')
    # print(dist_dict)
    return dist_dict

def get_admin_info(
    originalDf
):
    admin_list = originalDf["관리계정"].drop_duplicates(keep="first").to_list()
    admin_dict = {}
    # print(len(admin_list))
    for i in range(len(admin_list)):
        admin_dict[admin_list[i]] = i

    admin_dict_df = pd.DataFrame(data=admin_dict, index=[0])
    admin_dict_df = (admin_dict_df.T)
    admin_dict_df.to_excel('admin_dict.xlsx')
    # print(dist_dict)
    return admin_dict


"""
To-do
1. Inplacing dist-accnt with that from dist_dict
2. Have that to be a copy of originalDf
"""
def prepare_samilCoA(file_name: str, data_path: str, preprocess_type: str):
    # comp, abs, plain, part
    # This data will have headers as mentioned below
    # headers = ["계정코드", "회사계정", "1차번역", "계정과목", "관리계정", "공시용계정", "합산계정", "분류", "구분", "회사명", ]
    company_admin_headers = ["계정코드", "1차번역", "관리계정"]

    comp_admin_dis_headers = ["comparative_pos", "관리계정", "공시용계정", "회사명"]
    abs_admin_dis_headers = ["index", "관리계정", "공시용계정", "회사명"]
    plain_admin_dis_headers = ["계정코드", "관리계정", "공시용계정", "회사명"]
    part_admin_dis_headers = ["계정코드", "관리계정", "공시용계정", "회사명"]
    admin_dis_headers = ["관리계정", "공시용계정", "회사명"]

    comp_company_admin_headers = ["comparative_pos", "1차번역", "관리계정", "회사명"]
    abs_company_admin_headers = ["index", "1차번역", "관리계정", "회사명"]
    plain_company_admin_headers = ["계정코드", "1차번역", "관리계정", "회사명"]
    part_company_admin_headers = ["계정코드", "1차번역", "관리계정", "회사명"]
    company_admin_headers = ["1차번역", "관리계정", "회사명"]
    
    originalDf = pd.read_excel(os.path.join(data_path, file_name))
    originalDf["공시용계정"] = originalDf["공시용계정"].str.replace(' ', '')
    originalDf["계정과목"] = originalDf["계정과목"].str.replace(' ', '')
    originalDf["관리계정"] = originalDf["관리계정"].str.replace(' ', '')

    # List-ified independent distribution accounts
    # Enumerating dist-accnt by using .index will do the trick

    # inplacing dist-accnt with that from dist_dict
    copiedDf = originalDf.copy()
    # copiedDf["공시용계정"] = copiedDf["공시용계정"].map(dist_dict)

    # drop_duplicates?  

    # Usage of side information will be dealt in main.py
    if preprocess_type == "company_admin":
        copiedDf.loc[:,company_admin_headers].to_excel(os.path.join(data_path, "company_admin_df.xlsx"), index=False)
    elif preprocess_type == "comp_admin_dis":
        copiedDf.loc[:,comp_admin_dis_headers].to_excel(os.path.join(data_path, "comp_admin_dis_df.xlsx"), index=False)
    elif preprocess_type == "abs_admin_dis":
        copiedDf.loc[:,abs_admin_dis_headers].to_excel(os.path.join(data_path, "abs_admin_dis_df.xlsx"), index=False)
    elif preprocess_type == "plain_admin_dis":
        copiedDf.loc[:,plain_admin_dis_headers].to_excel(os.path.join(data_path, "plain_admin_dis_df.xlsx"), index=False)
    elif preprocess_type == "part_admin_dis":
        copiedDf.loc[:,part_admin_dis_headers].to_excel(os.path.join(data_path, "part_admin_dis_df.xlsx"), index=False)
    elif preprocess_type == "admin_dis":
        copiedDf.loc[:,admin_dis_headers].to_excel(os.path.join(data_path, "admin_dis_df.xlsx"), index=False)

    elif preprocess_type == "comp_company_admin":
        copiedDf.loc[:,comp_company_admin_headers].to_excel(os.path.join(data_path, "comp_company_admin_df.xlsx"), index=False)
    elif preprocess_type == "abs_company_admin":
        copiedDf.loc[:,abs_company_admin_headers].to_excel(os.path.join(data_path, "abs_company_admin_df.xlsx"), index=False)
    elif preprocess_type == "plain_company_admin":
        copiedDf.loc[:,plain_company_admin_headers].to_excel(os.path.join(data_path, "plain_company_admin_df.xlsx"), index=False)
    elif preprocess_type == "part_company_admin":
        copiedDf.loc[:,part_company_admin_headers].to_excel(os.path.join(data_path, "part_company_admin_df.xlsx"), index=False)
    elif preprocess_type == "company_admin":
        copiedDf.loc[:,company_admin_headers].to_excel(os.path.join(data_path, "company_admin_df.xlsx"), index=False)

if __name__ == "__main__":
    file_name = "SamilCoA2023(3).xlsx"
    data_name = "SamilCoA2023"
    data_path = os.path.join(BASE_PATH, data_name)

    parser = ArgumentParser(description="Preprocess CoA dataset")
    parser.add_argument("--preprocess_type", type=str, default="admin_dis")
    args = parser.parse_args()

    originalDf = pd.read_excel(os.path.join(data_path, file_name))
    originalDf["공시용계정"] = originalDf["공시용계정"].str.replace(' ', '')
    originalDf["계정과목"] = originalDf["계정과목"].str.replace(' ', '')
    originalDf["관리계정"] = originalDf["관리계정"].str.replace(' ', '')

    # Dist accnt needs to be enumerated
    dist_dict = get_dist_info(originalDf=originalDf)
    admin_dict = get_admin_info(originalDf=originalDf)

    # print(dist_dict)

    prepare_samilCoA(file_name=file_name, data_path=data_path, preprocess_type=args.preprocess_type)