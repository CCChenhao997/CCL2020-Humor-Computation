import re
import pandas as pd
from tqdm import tqdm


def data_process(input_path, output_path):
    
    df_train = pd.read_csv('data/raw/en_train.csv', encoding='utf-8')
    df_dev = pd.read_csv('data/raw/en_dev.csv', encoding='utf-8')
    train_list = df_train['Sentence'].tolist()
    dev_list = df_dev['Sentence'].tolist()
    raw_list = train_list + dev_list
    
    with open(input_path, "r") as f:
        idx_list, speaker_list, sentence_list = [], [], []
        idx = -1
        for i, line in tqdm(enumerate(f.readlines())):
            if ': ' in line and 'Scene' not in line and 'SCENE' not in line and line[0] not in ['(', '[']:
                speaker, _, sentence = line.partition(": ")
                sentence = re.sub('(\(.*?\))', '', sentence)
                if len(speaker.split()) <= 4 and 2 <= len(sentence.split()) <= 80 and \
                    sentence.strip() not in raw_list and sentence.strip() not in sentence_list:
                        
                    idx += 1
                    idx_list.append(idx)
                    speaker_list.append(speaker.strip())
                    sentence_list.append(sentence.strip())
        
    data = list(zip(idx_list, speaker_list, sentence_list))
    data_df = pd.DataFrame(data, columns=['ID', 'Speaker', 'Sentence'])
    data_df.to_csv(output_path, index=None, header=False, sep='\t')
    

if __name__ == "__main__":
    input_path = "data/stage_lines/Friends.txt"
    output_path = "data/stage_lines/Friends.tsv"
    data_process(input_path, output_path)