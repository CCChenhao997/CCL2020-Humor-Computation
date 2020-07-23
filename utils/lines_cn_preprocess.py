import pandas as pd
from tqdm import tqdm
import re 

def data_process(input_path, pre_output_path, output_path):
    # path = "data/stage_lines/Friends.txt"
    fw = open(pre_output_path, "w", encoding='utf-8')
    with open(input_path, "r", encoding='utf-8') as f:
        tmp_line = ''
        for line in tqdm(f.readlines()):
            if line[0] in ['(', '[', '（', '*', '=']:
                continue
            if not re.search('[\u4E00-\u9FA5]', line) or re.search('第.*集', line):
                continue
            if ':' in line or '：' in line and line[0] != ' ':
                if tmp_line:
                    # tmp_line = re.sub('[(（.*）)(\(.*))(（.*\))(\(.*）)([)]', '', tmp_line)
                    tmp_line = re.sub(u'（.*?）|\\(.*?\\)|（.*?\\)|\\(.*?）|\\[', '', tmp_line)
                    fw.write(tmp_line.strip() + '\n')
                    tmp_line = line.strip()
                else:
                    tmp_line = line.strip()
            else:
                if '***' in line or '===' in line:
                     continue
                tmp_line += line.strip()
    fw.close()

    df_train = pd.read_csv('data/raw/cn_train.csv', encoding='utf-8')
    df_dev = pd.read_csv('data/raw/cn_dev.csv', encoding='utf-8')
    train_list = df_train['Sentence'].tolist()
    dev_list = df_dev['Sentence'].tolist()
    raw_list = train_list + dev_list
    
    with open(pre_output_path, "r", encoding='utf-8') as f:
        idx_list, speaker_list, sentence_list = [], [], []
        idx = -1
        for i, line in tqdm(enumerate(f.readlines())):
            if '：' in line or ':' in line:
                # print(line)
                segments = re.split('[:：]', line.strip())
                speaker, sentence = segments[0], segments[1]
                if speaker == "简介":
                    continue 
                if len(speaker.split()) <= 5 and len(sentence.split()) <= 120 and sentence.strip() not in raw_list:
                    sentence = sentence.strip()
                    filtered = False
                    if len(sentence)>6:
                        #print(sentence)
                        for i in raw_list:
                            if sentence[:6] in i:
                                filtered = True
                                break
                    if filtered == False:
                        idx += 1
                        idx_list.append(idx)
                        speaker_list.append(speaker.strip())
                        sentence_list.append(sentence.strip())
        
    data = list(zip(idx_list, speaker_list, sentence_list))
    data_df = pd.DataFrame(data, columns=['ID', 'Speaker', 'Sentence'])
    data_df.to_csv(output_path, index=None)

if __name__ == "__main__":
    input_path = "data\我爱我家全台词.txt"
    pre_output_path = "data\wash_dict.txt"
    output_path = "data\我爱我家全台词.csv"
    data_process(input_path, pre_output_path, output_path)