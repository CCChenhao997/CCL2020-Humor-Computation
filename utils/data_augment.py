import os
import pandas as pd
from pprint import pprint
from textattack.augmentation import EasyDataAugmenter

fold_path = '/data/tmp/ch/CCL2020-Humor-Computation/data/data_StratifiedKFold_666/en/'
en_fold = os.listdir(fold_path)
# print(en_fold[0] + 'train.csv')

augmenter = EasyDataAugmenter(alpha=0.2, n_aug=1)

for i, fold in enumerate(en_fold):
    print("第{}个，进行中。。。".format(i))
    data_path = fold_path + fold + '/train.csv'
    df = pd.read_csv(data_path, sep='\t', header=None, encoding='utf-8', engine='python')
    dialogue_id = []
    speaker = []
    sentence = []
    polarity = []
    for index, line in df.iterrows():
        try:
            if int(line[3]) == 1:
                sentence.append(augmenter.augment(line[2])[0])
            else:
                continue
            dialogue_id.append(line[0])
            speaker.append(line[1])
            polarity.append(int(line[3]))
        except:
            continue
        
    augment_data = list(zip(dialogue_id, speaker, sentence, polarity))
    df_augment = pd.DataFrame(augment_data)

    output_path = './{}_aug.csv'.format(fold)
    df_augment.to_csv(output_path, sep='\t', index = False, header = False, encoding='utf-8')