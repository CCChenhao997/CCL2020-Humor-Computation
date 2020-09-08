import pandas as pd



def filter_dif(simple_path, prob_path, last_vote_path, dev_data_path, save_path, dif_save_path):
    simple_label = pd.read_csv(simple_path, encoding='utf-8')
    prob_label = pd.read_csv(prob_path, encoding='utf-8')
    last_label = pd.read_csv(last_vote_path, encoding='utf-8')

    dev_data = pd.read_csv(dev_data_path, encoding='utf-8')
    dev_data['simple_label'] = simple_label[['Label']]
    dev_data['prob_label'] = prob_label[['Label']]
    dev_data['last_label'] = last_label[['Label']]

    dev_data['same'] = dev_data.apply(lambda x: x['simple_label'] == x['prob_label'] and x['simple_label'] == x['last_label'], axis = 1)
    
    dev_data.to_csv(save_path, sep='\t', index=False, header=True, \
                columns=['Dialogue_id', 'Speaker', 'Sentence', 'simple_label', 'prob_label', 'last_label', 'same'], encoding='utf-8')
    dev_data = dev_data[dev_data['same'] == False] 
    dev_data.to_csv(dif_save_path, sep='\t', index=False, header=True, \
                columns=['Dialogue_id', 'Speaker', 'Sentence', 'simple_label', 'prob_label', 'last_label', 'same'], encoding='utf-8')
    print((dev_data[['simple_label']] == 1).sum())
    print((dev_data[['prob_label']] == 1).sum())
    print((dev_data[['last_label']] == 1).sum())
    




simple_path = r'./predict_data/第十周/bert_mixed_0823/vote/bert_mixed-cn-voted.csv'
prob_path = r'./predict_data/第十周/bert_mixed_0823/vote/bert_mixed-cn-prob_voted.csv'
last_vote_path = r'./predict_data/第九周/cn/bert_spc_rev-cn-voted_simple.csv'
dev_data_path = r'./data/raw/cn_dev.csv'
save_path = r'./predict_data/第十周/bert_mixed_0823/vote/dif-voted3.csv'
dif_save_path = r'./predict_data/第十周/bert_mixed_0823/vote/only-dif-voted3.csv'

# simple_path = r'./predict_data/第十周/bert_spc_rev_0822/5-fold-1/vote/bert_spc_rev-en-simple-voted.csv'
# prob_path = r'./predict_data/第十周/bert_spc_rev_0822/5-fold-1/vote/bert_spc_rev-en-prob-voted.csv'
# last_vote_path = r'./predict_data/第九周/bert_spc_rev_0816/vote/bert_spc_rev-en-simple-voted.csv'
# dev_data_path = r'./data/raw/en_dev.csv'
# save_path = r'./predict_data/第十周/bert_spc_rev_0822/5-fold-1/vote/dif-voted3.csv'
# dif_save_path = r'./predict_data/第十周/bert_spc_rev_0822/5-fold-1/vote/only-dif-voted3.csv'


filter_dif(simple_path, prob_path, last_vote_path, dev_data_path, save_path, dif_save_path)