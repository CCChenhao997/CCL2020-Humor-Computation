import os


with open('./spider/summary.txt', 'r') as sample:
    f = open('./spider/summary_pretrain.txt','w+')
    for line in sample.readlines():
        if ':' not in line:
            continue
        curLine = line.strip().split(":")
        speaker = curLine[0].strip()
        sentence = curLine[1].strip()
        # if len(speaker.split()) > 10:
        #     print(speaker)
        if len(speaker.split()) < 1:
            print(speaker)
        if len(sentence.split()) < 1:
            continue
        f.write(speaker+'\r\n')
        f.write(sentence+'\r\n')
        f.write('\r\n')
    f.close()

print("写入成功")