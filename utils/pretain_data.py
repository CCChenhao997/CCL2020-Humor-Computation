import pandas as pd

path = "../data/preprocess/en_test.tsv"
df = pd.read_csv(path, sep='\t', header=None)

with open('../data/pretrain/en_pretrain_test.txt','w+') as f:
    for index, line in df.iterrows():
        dialogue_id = int(line[0])
        speaker = line[1].strip()
        sentence = line[2].strip()
        # f.writelines(speaker)
        f.write(speaker+'\r\n')
        f.write(sentence+'\r\n')
        f.write('\r\n')