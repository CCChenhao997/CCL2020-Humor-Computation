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
    date = '0725'
    dataset = 'en'
    pseudo = False

    if pseudo:
        DATA_DIR = '../predict_data/aug_data/{}_{}_pseudo/{}/'.format(model_name, date, dataset)
    else:
        DATA_DIR = '../predict_data/aug_data/{}_{}/{}/'.format(model_name, date, dataset)
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
    
    df_data = pd.read_csv('../data/preprocess/en_aug.tsv', sep='\t', names=["ID", "Speaker", "Sentence"])
    ID_list = [i for i in range(df_data.shape[0])]
    df_data['ID'] = pd.Series(ID_list)

    # pprint(df_data.head(5))
    df_merged = df_merged.merge(df_data, how='left', on='ID')
    # pprint(df_merged.shape)

    speaker_list, sentence_list, label_list = [], [], []
    for index, line in df_merged.iterrows():
        label_1 = int(line[1])
        label_2 = int(line[2])
        label_3 = int(line[3])
        label_4 = int(line[4])
        label_5 = int(line[5])
        speaker = line[6]
        sentence = line[7]
        label = None
        if label_1 + label_2 + label_3 + label_4 + label_5 >= 3:
            label = 1
        elif label_1 == label_2 == label_3 == label_4 == label_5 == 0:
            label = 0

        if label is not None:
            speaker_list.append(speaker)
            sentence_list.append(sentence)
            label_list.append(label)
    
    print(len(speaker_list), len(sentence_list), len(label_list))
    idx_list = [i for i in range(len(speaker_list))]
    # final_data = list(zip(idx_list, speaker_list, sentence_list, label_list))
    # final_data = pd.DataFrame(final_data, columns=['ID', 'Speaker', 'Sentence', 'Label'])
    final_data = list(zip(idx_list, idx_list, idx_list, speaker_list, sentence_list, label_list))
    final_data = pd.DataFrame(final_data, columns=['ID', 'Dialogue_id', 'Utterance_id', 'Speaker', 'Sentence', 'Label'])
    

    if pseudo:
        save_path = '../predict_data/aug_data/{}_{}_pseudo/vote'.format(model_name, date)
    else:
        save_path = '../predict_data/aug_data/{}_{}/vote'.format(model_name, date)
    if not os.path.exists(save_path):
        os.makedirs(save_path, mode=0o777)
    
    file_path = '{}/{}-{}-voted.csv'.format(save_path, model_name, dataset)
    # final_data.to_csv(file_path, index=None, header=None, sep='\t')
    final_data.to_csv(file_path, index=None)
    print("写入成功!")