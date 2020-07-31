import os
import pandas as pd

# path = "../data/preprocess/en_test.tsv"
# df = pd.read_csv(path, sep='\t', header=None)

# with open('../data/pretrain/en_pretrain_test.txt','w+') as f:
#     for index, line in df.iterrows():
#         dialogue_id = int(line[0])
#         speaker = line[1].strip()
#         sentence = line[2].strip()
#         f.write(speaker+'\r\n')
#         f.write(sentence+'\r\n')
#         f.write('\r\n')

rootdir = '../data/dialog'
file_list = os.listdir(rootdir)  # 列出文件夹下所有的目录与文件
with open('../data/pretrain/cn/cn_pretrain_dialogue.txt','w+') as f:
    for i in range(0, len(file_list)):
        path = os.path.join(rootdir, file_list[i])
        if os.path.isfile(path):
            with open(path, 'r') as dig:
                for line in dig.readlines():
                    curLine = line.strip().split(":")
                    speaker = curLine[0].strip()
                    sentence = curLine[1].strip()
                    if len(speaker) > 5 or 'http' in speaker:
                        continue
                    if len(sentence) < 5 or 'http' in sentence:
                        continue
                    f.write(speaker+'\r\n')
                    f.write(sentence+'\r\n')
                    f.write('\r\n')

print("写入成功")