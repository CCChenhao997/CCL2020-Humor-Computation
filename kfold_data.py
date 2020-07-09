import os
import pandas as pd
import numpy as np
from collections import Counter
from sklearn.model_selection import StratifiedKFold, KFold
from imblearn.under_sampling import RandomUnderSampler



def generate_data_imb(random_state=666, dataset=None, date=None):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

    for fold, (train_index, dev_index) in enumerate(skf.split(X, y)):
        print(fold, "TRAIN:", train_index, "TEST:", dev_index)

        DATA_DIR = "./data/data_StratifiedKFold_{}_{}/{}/data_fold_{}/".format(random_state, date, dataset, fold)
        if not os.path.exists(DATA_DIR):
            os.makedirs(DATA_DIR)
        tmp_train_df = train_df.iloc[train_index]
        tmp_dev_df = train_df.iloc[dev_index]
        print(len(train_index))
        # print(type(train_index))
        print(tmp_train_df['Label'].value_counts())
        
        rus = RandomUnderSampler(random_state=random_state)
        k_X = train_index.reshape(-1, 1)
        # print(type(k_X))
        k_y = tmp_train_df.loc[:, 'Label'].to_numpy()

        X_resampled, y_resampled = rus.fit_resample(k_X, k_y)
        print("X_resampled", X_resampled)
        print(len(X_resampled))
        print("y_resampled", y_resampled)
        print(len(y_resampled))
        print(sorted(Counter(y_resampled).items()))
        exit()
        tmp_train_df.to_csv(DATA_DIR + "/train.csv", index=False, header=False, sep='\t', encoding='utf-8')
        tmp_dev_df.to_csv(DATA_DIR + "/test.csv", index=False, header=False, sep='\t', encoding='utf-8')
        print(tmp_train_df.shape, tmp_dev_df.shape)

def generate_data(random_state=666, dataset=None, date=None, under_sampling = True):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)  # ! 自动按比例划分？

    for fold, (train_index, dev_index) in enumerate(skf.split(X, y)):
        print(fold, "TRAIN:", train_index, "TEST:", dev_index)
        DATA_DIR = "./data/data_StratifiedKFold_{}_{}/{}/data_fold_{}/".format(random_state, date, dataset, fold)
        if not os.path.exists(DATA_DIR):
            os.makedirs(DATA_DIR)
        tmp_train_df = train_df.iloc[train_index]
        tmp_dev_df = train_df.iloc[dev_index]
        
        if under_sampling:
            print(tmp_train_df['Label'].value_counts())
            pos_df = tmp_train_df[tmp_train_df.loc[:, 'Label'] == 1]
            neg_df = tmp_train_df[tmp_train_df.loc[:, 'Label'] == 0]

            del_neg_df = neg_df.sample(n = 1000, replace = False, random_state = random_state)
            res_df_idx = list(set(list(neg_df.index)) - set(list(del_neg_df.index)))
            # print(list(pos_df.index))
            res_df_idx = sorted(res_df_idx + list(pos_df.index))
            tmp_train_df = train_df.iloc[np.array(res_df_idx)]
            print(tmp_train_df['Label'].value_counts())
        
        tmp_train_df.to_csv(DATA_DIR + "/train.csv", index=False, header=False, sep='\t', encoding='utf-8')
        tmp_dev_df.to_csv(DATA_DIR + "/test.csv", index=False, header=False, sep='\t', encoding='utf-8')
        print(tmp_train_df.shape, tmp_dev_df.shape)



if __name__=="__main__":
    dataset = 'en'
    date = '0709'
    # path = './data/pseudo_{}/{}_total_pseudo.csv'.format(date, dataset)
    # path =  './data/transform/{}_total.csv'.format(dataset)
    train_df = pd.read_csv('data/raw/en_train.csv', encoding='utf-8')
    # train_df = pd.read_csv(path, encoding='utf-8')
    train_df = train_df[['Dialogue_id', 'Speaker', 'Sentence', 'Label']]
    # train_df = train_df[['Dialogue_id', 'Speaker1', 'Speaker2', 'Sentence1', 'Sentence2', 'Label']]
    X = np.array(train_df.index)
    y = train_df.loc[:, 'Label'].to_numpy()
    
    generate_data(random_state=666, dataset=dataset, date=date)