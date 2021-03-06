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


def simple_vote(model_name, date, dataset, pseudo=False):
    if pseudo:
        DATA_DIR = '../predict_data/{}_{}_pseudo/{}/'.format(model_name, date, dataset)
    else:
        DATA_DIR = '../predict_data/{}_{}/{}/'.format(model_name, date, dataset)
    files = os.listdir(DATA_DIR)
    files = [i for i in files]

    i = 0
    for fname in files:
        tmp_df = pd.read_csv(DATA_DIR + fname)
        tmp_df = pd.DataFrame(tmp_df, columns=['ID', 'Label'])
        if i == 0:
            df_merged = pd.read_csv(DATA_DIR + fname)
            df_merged = pd.DataFrame(df_merged, columns=['ID', 'Label'])
        if i > 0:
            df_merged = df_merged.merge(tmp_df, how='left', on='ID')
        print(df_merged.shape)
        i += 1

    tmp_label = np.array(df_merged.iloc[:, 1:])
    voted_label = [work(line) for line in tmp_label]
    df_summit = df_merged[['ID']]
    df_summit = df_summit.copy()
    df_summit['Label'] = voted_label

    if pseudo:
        save_path = '../predict_data/{}_{}_pseudo/vote'.format(model_name, date)
    else:
        save_path = '../predict_data/{}_{}/vote'.format(model_name, date)
    if not os.path.exists(save_path):
        os.makedirs(save_path, mode=0o777)
    
    file_path = '{}/{}-{}-voted.csv'.format(save_path, model_name, dataset)
    df_summit.to_csv(file_path, index=None)
    print("写入成功!")


def aug_vote(model_name, date, dataset, pseudo=False):
    if pseudo:
        DATA_DIR = '../predict_data/aug_data/{}_{}_pseudo/{}/'.format(model_name, date, dataset)
    else:
        DATA_DIR = '../predict_data/aug_data/{}_{}/{}/'.format(model_name, date, dataset)
    files = os.listdir(DATA_DIR)
    files = [i for i in files]
    i = 0
    for fname in files:
        tmp_df = pd.read_csv(DATA_DIR + fname)
        tmp_df = pd.DataFrame(tmp_df, columns=['ID', 'Label'])
        if i == 0:
            df_merged = pd.read_csv(DATA_DIR + fname)
            df_merged = pd.DataFrame(df_merged, columns=['ID', 'Label'])
        if i > 0:
            df_merged = df_merged.merge(tmp_df, how='left', on='ID')
        print(df_merged.shape)
        i += 1
 
    df_data = pd.read_csv('../data/test_data/cn_test.csv', sep=',')
    # df_data = pd.read_csv('../data/test_data/cn_test.csv', sep='\t', names=["ID", "Speaker", "Sentence"])
    ID_list = [i for i in range(df_data.shape[0])]
    df_data['ID'] = pd.Series(ID_list)
    df_merged = df_merged.merge(df_data, how='left', on='ID')
    speaker_list, sentence_list, label_list = [], [], []
    humor_speaker_list, humor_sentence_list, humor_label_list = [], [], []
    un_speaker_list, un_sentence_list, un_label_list = [], [], []
    for index, line in df_merged.iterrows():
        label_1 = int(line[1])
        label_2 = int(line[2])
        label_3 = int(line[3])
        label_4 = int(line[4])
        label_5 = int(line[5])
        speaker = line[8]
        sentence = line[9]
        label = None
        if label_1 + label_2 + label_3 + label_4 + label_5 == 5:
            label = 1
            humor_speaker_list.append(speaker)
            humor_sentence_list.append(sentence)
            humor_label_list.append(label)
        elif label_1 == label_2 == label_3 == label_4 == label_5 == 0:
            label = 0
            un_speaker_list.append(speaker)
            un_sentence_list.append(sentence)
            un_label_list.append(label)

        if label is not None:
            speaker_list.append(speaker)
            sentence_list.append(sentence)
            label_list.append(label)

    
    print(len(speaker_list), len(sentence_list), len(label_list))
    print(len(humor_speaker_list), len(humor_sentence_list), len(humor_label_list))
    print(len(un_speaker_list), len(un_sentence_list), len(un_label_list))

    idx_list = [i for i in range(len(speaker_list))]
    humor_idx_list = [i for i in range(len(humor_speaker_list))]
    un_idx_list = [i for i in range(len(un_speaker_list))]


    # * tsv格式
    final_data = list(zip(idx_list, speaker_list, sentence_list, label_list))
    final_data = pd.DataFrame(final_data, columns=['ID', 'Speaker', 'Sentence', 'Label'])

    humor_final_data = list(zip(humor_idx_list, humor_speaker_list, humor_sentence_list, humor_label_list))
    humor_final_data = pd.DataFrame(humor_final_data, columns=['ID', 'Speaker', 'Sentence', 'Label'])

    un_final_data = list(zip(un_idx_list, un_speaker_list, un_sentence_list, un_label_list))
    un_final_data = pd.DataFrame(un_final_data, columns=['ID', 'Speaker', 'Sentence', 'Label'])

    # * csv格式
    # final_data = list(zip(idx_list, idx_list, idx_list, speaker_list, sentence_list, label_list))
    # final_data = pd.DataFrame(final_data, columns=['ID', 'Dialogue_id', 'Utterance_id', 'Speaker', 'Sentence', 'Label'])
    
    if pseudo:
        save_path = '../predict_data/aug_data/{}_{}_pseudo/vote'.format(model_name, date)
    else:
        save_path = '../predict_data/aug_data/{}_{}/vote'.format(model_name, date)
        humor_save_path = '../predict_data/aug_data/{}_{}/humor_vote'.format(model_name, date)
        un_save_path = '../predict_data/aug_data/{}_{}/un_vote'.format(model_name, date)

    if not os.path.exists(save_path):
        os.makedirs(save_path, mode=0o777)
        os.makedirs(humor_save_path, mode=0o777)
        os.makedirs(un_save_path, mode=0o777)
    
    file_path = '{}/{}-{}-voted.tsv'.format(save_path, model_name, dataset)
    humor_file_path = '{}/{}-{}-voted.tsv'.format(humor_save_path, model_name, dataset)
    un_file_path = '{}/{}-{}-voted.tsv'.format(un_save_path, model_name, dataset)

    # * tsv格式
    final_data.to_csv(file_path, index=None, header=None, sep='\t')
    humor_final_data.to_csv(humor_file_path, index=None, header=None, sep='\t')
    un_final_data.to_csv(un_file_path, index=None, header=None, sep='\t')

    # * csv格式
    # final_data.to_csv(file_path, header=None)
    # humor_final_data.to_csv(humor_file_path, header=None)
    # un_final_data.to_csv(un_file_path, header=None)
    print("写入成功!")


def prob_vote(model_name, date, dataset, pseudo=False):
    if pseudo:
        DATA_DIR = '../predict_data/{}_{}_pseudo/{}/'.format(model_name, date, dataset)
    else:
        DATA_DIR = '../predict_data/{}_{}/{}/'.format(model_name, date, dataset)
    files = os.listdir(DATA_DIR)
    files = [i for i in files]
    i = 0
    for fname in files:
        tmp_df = pd.read_csv(DATA_DIR + fname)
        if i == 0:
            df_merged = pd.read_csv(DATA_DIR + fname)
            df_prob_0 = df_merged['Prob_0']
            df_prob_1 = df_merged['Prob_1']
        else:
            df_prob_0 = df_prob_0.add(tmp_df['Prob_0'])
            df_prob_1 = df_prob_1.add(tmp_df['Prob_1'])
        i += 1
    
    label = []
    df_prob_0 = df_prob_0.tolist()
    df_prob_1 = df_prob_1.tolist()
 
    for prob_0, prob_1 in zip(df_prob_0, df_prob_1):
        if prob_1 >= prob_0:
            label.append(1)
        else:
            label.append(0)
    id = [i for i in range(len(label))]
    vote_results = pd.DataFrame(list(zip(id, label)))

    if pseudo:
        save_path = '../predict_data/{}_{}_pseudo/vote'.format(model_name, date)
    else:
        save_path = '../predict_data/{}_{}/vote'.format(model_name, date)
    if not os.path.exists(save_path):
        os.makedirs(save_path, mode=0o777)
    
    file_path = '{}/{}-{}-voted.csv'.format(save_path, model_name, dataset)
    vote_results.to_csv(file_path, index=None, header=['ID', 'Label'])
    print("写入成功!")


if __name__=="__main__":

    model_name = 'bert_spc_rev'
    date = '0824'
    dataset = 'cn'
    pseudo = False
    # simple_vote(model_name, date, dataset, pseudo)
    prob_vote(model_name, date, dataset, pseudo)
    # aug_vote(model_name, date, dataset, pseudo)
    