import pandas as pd

path = './predict_data/aug_data/bert_spc_cnn_0718/vote/bert_spc_cnn-en-voted.csv'
df = pd.read_csv(path, sep='\t', header=None, encoding='utf-8', engine='python')
# print(df.head(5))

# 0    27023
# 1    17531
# print(df[3].value_counts())

df_positive = df[df[3]==1]
# print(df_positive.shape)
# print(df_positive.head(5))

df_positive_sample = df_positive.sample(n=3000, random_state=666)
print(df_positive_sample.shape)

out_path = './predict_data/aug_data/bert_spc_cnn_0718/vote/bert_spc_cnn-en-voted_sample.csv'
df_positive_sample.to_csv(out_path, sep='\t', index=None, header=None)