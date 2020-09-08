import copy
import pandas as pd
from pprint import pprint




#! en/cn
dataset = 'en'
date = '0819'
# path = 'bert_spc_lay_0711'

# * 读取五折模型投票的结果
# pseudo_label_path = './predict_data/{}/vote/bert_spc-{}-voted.csv'.format(path, dataset)
pseudo_label_path = './predict_data/第九周/bert_spc_rev_0816/vote/bert_spc_rev-en-simple-voted.csv'  # ! 更换
pseudo_label = pd.read_csv(pseudo_label_path, encoding='utf-8')

# * 读取开发集数据，并打上伪标签
dev_data_path = './data/raw/{0}_dev.csv'.format(dataset)
dev_data = pd.read_csv(dev_data_path, encoding='utf-8')
dev_data['Label'] = pseudo_label[['Label']]
pseudo_data = dev_data.copy()
# pprint(pseudo_data.head(5))
pseudo_data_save_path = './data/pseudo_{}/{}_pseudo.csv'.format(date, dataset)
pseudo_data.to_csv(pseudo_data_save_path, index=False, encoding='utf-8')
# pseudo_data.to_csv(pseudo_data_save_path, sep='\t', index=False, header=False, \
#                     columns=['Dialogue_id', 'Speaker', 'Sentence', 'Label'], encoding='utf-8')


# # * 读取训练集数据，添加上伪标签的开发集数据
# train_data_path = './data/raw/{0}_train.csv'.format(dataset)
# train_data = pd.read_csv(train_data_path, encoding='utf-8')

# # pseudo_data_ = pseudo_data[['Dialogue_id', 'Speaker', 'Sentence', 'Label']]
# # train_data.columns = ['Dialogue_id', 'Speaker', 'Sentence', 'Label']
# df = pd.concat([train_data, pseudo_data])
# train_data_save_path = './data/pseudo_{}/{}_total_pseudo.csv'.format(date, dataset)
# df.to_csv(train_data_save_path, index=False, encoding='utf-8')

