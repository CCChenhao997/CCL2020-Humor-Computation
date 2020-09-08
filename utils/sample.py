import pandas as pd

# path = './predict_data/第五周/aug_data/bert_spc_cnn_0718/vote/bert_spc_cnn-en-voted.csv'
path = '../predict_data/aug_data/bert_spc_rev_0824/vote/bert_spc_rev-cn-voted.tsv'
df = pd.read_csv(path, sep='\t', header=None, encoding='utf-8', engine='python')
# df = pd.read_csv(path, encoding='utf-8', header=None, engine='python')
# print(df.head(5))

# 1    20648
# 0    16005
# print(df['Label'].value_counts())
# print(df[4].value_counts())
# * cn_test
# 0    2692
# 1     722

# df_positive = df[df['Label']==1]
# df_positive = df[df[3]==1]

df_positive_sample = df.sample(n=500, random_state=666)
# print(df_positive_sample.head())

# out_path = '../predict_data/第五周/aug_data/bert_mixed_0720/vote/bert_mixed-cn-voted_sample.csv'
out_path = '../predict_data/aug_data/bert_spc_rev_0824/vote/bert_spc_rev-cn-voted_sample_500.tsv'

df_positive_sample.to_csv(out_path, sep='\t', index=None, header=None)
# df_positive_sample.to_csv(out_path, index=None)