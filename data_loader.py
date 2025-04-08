import pandas as pd

def add_ids(data_dict):
    for idx, data in enumerate(data_dict):
        data["id"] = idx
    return data_dict

def load_data_as_dict(path):
    ""
    df = pd.read_csv(path)
    df = df[["normalized_text"]]
    df_dict = df.to_dict(orient="records")
    df_dict = add_ids(data_dict=df_dict)

    return df_dict
