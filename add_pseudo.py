import copy
import pandas as pd
from pprint import pprint

#! en/cn
dataset = 'cn'
pseudo_label_path = './predict_data/vote/bert_spc-{0}-voted.csv'.format(dataset)
pseudo_label = pd.read_csv(pseudo_label_path, encoding='utf-8')

raw_data_path = './data/raw/{0}_dev.csv'.format(dataset)
raw_data = pd.read_csv(raw_data_path, encoding='utf-8')
raw_data['Label'] = pseudo_label[['Label']]
pseudo_data = raw_data.copy()
# pprint(pseudo_data.head(5))

pseudo_data_save_path = './data/pseudo/{0}_pseudo.csv'.format(dataset)
pseudo_data.to_csv(pseudo_data_save_path, index=False, encoding='utf-8')
# pseudo_data.to_csv(pseudo_data_save_path, index=False, header=False,
#                 columns=['Dialogue_id', 'Speaker', 'Sentence', 'Label'], sep='\t', encoding='utf-8')

train_data_path = './data/raw/{0}_train.csv'.format(dataset)
train_data = pd.read_csv(train_data_path, encoding='utf-8')
# pprint(train_data.head(5))

# pseudo_data_ = pseudo_data[['Dialogue_id', 'Speaker', 'Sentence', 'Label']]
# train_data.columns = ['Dialogue_id', 'Speaker', 'Sentence', 'Label']
df = pd.concat([train_data, pseudo_data])
train_data_save_path = './data/pseudo/{0}_total_pseudo.csv'.format(dataset)
df.to_csv(train_data_save_path, index=False, encoding='utf-8')

# print(pseudo_data_.shape)   # (914, 4)
# print(train_data.shape)     # (7472, 4)
# print(df.shape)             # (8386, 4)