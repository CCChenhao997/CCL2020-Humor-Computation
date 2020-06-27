import os
import pandas as pd
import numpy as np
import copy
from pprint import pprint


def work(pres):
    count = [0, 0]
    for i in pres:
        count[i] += 1
    out = count.index(max(count))
    return out


if __name__=="__main__":

    model_name = 'bert_spc'
    date = '0627'
    dataset = 'cn'
    DATA_DIR = './predict_data/{}_{}/{}/'.format(model_name, date, dataset)
    files = os.listdir(DATA_DIR)
    files = [i for i in files]

    i = 0
    for fname in files:
        tmp_df = pd.read_csv(DATA_DIR + fname)
        if i == 0:
            df_merged = pd.read_csv(DATA_DIR + fname)
        if i > 0:
            df_merged = df_merged.merge(tmp_df, how='left', on='ID')
        print(df_merged.shape)
        i += 1

    # pprint(df_merged.head(5))
    tmp_label = np.array(df_merged.iloc[:, 1:])
    voted_label = [work(line) for line in tmp_label]
    # print(voted_label)
    df_summit = df_merged[['ID']]
    df_summit = df_summit.copy()
    df_summit['Label'] = voted_label
    # pprint(df_summit.head(5))
    sava_path = './predict_data/{}_{}/vote/{}-{}-voted.csv'.format(model_name, date, model_name, dataset)
    df_summit.to_csv(sava_path, index=None)