import pandas as pd

# path = './predict_data/第五周/aug_data/bert_spc_cnn_0718/vote/bert_spc_cnn-en-voted.csv'
path = '../predict_data/aug_data/bert_spc_0730/vote/bert_spc-en-voted.csv'
# df = pd.read_csv(path, sep='\t', header=None, encoding='utf-8', engine='python')
df = pd.read_csv(path, encoding='utf-8', engine='python')
# print(df.head(5))

# 1    20648
# 0    16005
print(df['Label'].value_counts())
# print(df[3].value_counts())

df_positive = df[df['Label']==0]
# df_positive = df[df[3]==1]

df_positive_sample = df_positive.sample(n=500, random_state=666)

# out_path = '../predict_data/第五周/aug_data/bert_mixed_0720/vote/bert_mixed-cn-voted_sample.csv'
out_path = '../predict_data/aug_data/bert_spc_0730/vote/bert_spc-en-voted_sample_500.csv'

# df_positive_sample.to_csv(out_path, sep='\t', index=None, header=None)
df_positive_sample.to_csv(out_path, index=None)