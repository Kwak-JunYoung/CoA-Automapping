import os
from pandas import pd
from argparse import ArgumentParser

BASE_PATH = "./data"

# 공시용계정과, 그의 번호
def get_dist_info(
    data_path,
    file_name
):
    df_dis = pd.read_excel(os.path.join(data_path, file_name), usecols=["공시용계정"]).drop_na(inplace=True).drop_duplicates(subset=["공시용계정"], keep="first", inplace=True)
    df_dis["idx"] = [i for i in range(len(df_dis))]
    print(df_dis)
    return df_dis

def prepare_samilCoA(data_name: str, preprocess_type: str):
    # comp, abs, plain, part
    # This data will have headers as mentioned below
    file_name = "SamilCoA2023(2).xlsx"
    data_name = "SamilCoA2023"
    data_path = os.path.join(BASE_PATH, data_name)
    # headers = ["계정코드", "회사계정", "1차번역", "계정과목", "관리계정", "공시용계정", "합산계정", "분류", "구분", "회사명", ]
    comp_admin_headers = ["계정코드", "1차번역", "관리계정"]

    comp_admin_dis_headers = ["comparative_pos", "관리계정", "공시용계정", "회사명"]
    abs_admin_dis_headers = ["index", "관리계정", "공시용계정", "회사명"]
    plain_admin_dis_headers = ["계정코드", "관리계정", "공시용계정", "회사명"]
    part_admin_dis_headers = ["계정코드", "관리계정", "공시용계정", "회사명"]

    if "admin_dis" in preprocess_type:
        df_dis = pd.read_excel(os.path.join(data_path, file_name), usecols=["공시용계정"]).drop_na(inplace=True).drop_duplicates(subset=["공시용계정"], keep="first", inplace=True)
    
    usecols = []

    if preprocess_type == "company_admin":
        usecols = comp_admin_headers
    elif preprocess_type == "abs_admin_dis":
        usecols = abs_admin_dis_headers
    elif preprocess_type == "comp_admin_dis":
        usecols = comp_admin_dis_headers
    elif preprocess_type == "plain_admin_dis":
        usecols = plain_admin_dis_headers
    elif preprocess_type == "part_admin_dis":
        usecols = part_admin_dis_headers                

    df = pd.read_excel(os.path.join(data_path, file_name), usecols=usecols)

    # drop_duplicates?

    # Dist accnt needs to be enumerated
    
    # Usage of side information will be dealt in main.py
    if preprocess_type == "company_admin":
        df.to_excel(os.path.join(data_path, "company_admin_df.xlsx"), index=False)
    elif preprocess_type == "comp_admin_dis":
        df.to_excel(os.path.join(data_path, "comp_admin_dis_df.xlsx"), index=False)
    elif preprocess_type == "abs_admin_dis":
        df.to_excel(os.path.join(data_path, "abs_admin_dis_df.xlsx"), index=False)
    elif preprocess_type == "plain_admin_dis":
        df.to_excel(os.path.join(data_path, "plain_admin_dis_df.xlsx"), index=False)
    elif preprocess_type == "part_admin_dis":
        df.to_excel(os.path.join(data_path, "part_admin_dis_df.xlsx"), index=False)


if __name__ == "__main__":
    file_name = "SamilCoA2023(2).xlsx"
    data_name = "SamilCoA2023"
    data_path = os.path.join(BASE_PATH, data_name)

    parser = ArgumentParser(description="Preprocess CoA dataset")
    parser.add_argument("--preprocess_type", type=str, default="admin_dis")
    args = parser.parse_args()

    prepare_samilCoA(data_name=data_name, preprocess_type=args.preprocess_type)