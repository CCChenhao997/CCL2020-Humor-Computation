import copy
import pandas as pd
from pprint import pprint

#! en/cn
dataset = 'cn'
date = '0704'
path = 'bert_spc_0704'

# * 读取五折模型投票的结果
pseudo_label_path = './predict_data/{}/vote/bert_spc-{}-voted.csv'.format(path, dataset)
pseudo_label = pd.read_csv(pseudo_label_path, encoding='utf-8')

# * 读取开发集数据，并打上伪标签
raw_data_path = './data/raw/{0}_dev.csv'.format(dataset)
raw_data = pd.read_csv(raw_data_path, encoding='utf-8')
raw_data['Label'] = pseudo_label[['Label']]
pseudo_data = raw_data.copy()
# pprint(pseudo_data.head(5))
pseudo_data_save_path = './data/pseudo_{}/{}_pseudo.csv'.format(date, dataset)
pseudo_data.to_csv(pseudo_data_save_path, index=False, encoding='utf-8')
# pseudo_data.to_csv(pseudo_data_save_path, index=False, header=False,
#                 columns=['Dialogue_id', 'Speaker', 'Sentence', 'Label'], sep='\t', encoding='utf-8')

# * 读取训练集数据，添加上伪标签的开发集数据
train_data_path = './data/raw/{0}_train.csv'.format(dataset)
train_data = pd.read_csv(train_data_path, encoding='utf-8')
# pprint(train_data.head(5))

# pseudo_data_ = pseudo_data[['Dialogue_id', 'Speaker', 'Sentence', 'Label']]
# train_data.columns = ['Dialogue_id', 'Speaker', 'Sentence', 'Label']
df = pd.concat([train_data, pseudo_data])
train_data_save_path = './data/pseudo_{}/{}_total_pseudo.csv'.format(date, dataset)
df.to_csv(train_data_save_path, index=False, encoding='utf-8')

# print(pseudo_data_.shape)   # (914, 4)
# print(train_data.shape)     # (7472, 4)
# print(df.shape)             # (8386, 4)
