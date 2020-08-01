import re
import time
import requests
import random
# from tqdm import tqdm

USER_AGENTS = [
    "Mozilla/4.0 (compatible; MSIE 6.0; Windows NT 5.1; SV1; AcooBrowser; .NET CLR 1.1.4322; .NET CLR 2.0.50727)",
    "Mozilla/4.0 (compatible; MSIE 7.0; Windows NT 6.0; Acoo Browser; SLCC1; .NET CLR 2.0.50727; Media Center PC 5.0; .NET CLR 3.0.04506)",
    "Mozilla/4.0 (compatible; MSIE 7.0; AOL 9.5; AOLBuild 4337.35; Windows NT 5.1; .NET CLR 1.1.4322; .NET CLR 2.0.50727)",
    "Mozilla/5.0 (Windows; U; MSIE 9.0; Windows NT 9.0; en-US)",
    "Mozilla/5.0 (compatible; MSIE 9.0; Windows NT 6.1; Win64; x64; Trident/5.0; .NET CLR 3.5.30729; .NET CLR 3.0.30729; .NET CLR 2.0.50727; Media Center PC 6.0)",
    "Mozilla/5.0 (compatible; MSIE 8.0; Windows NT 6.0; Trident/4.0; WOW64; Trident/4.0; SLCC2; .NET CLR 2.0.50727; .NET CLR 3.5.30729; .NET CLR 3.0.30729; .NET CLR 1.0.3705; .NET CLR 1.1.4322)",
    "Mozilla/4.0 (compatible; MSIE 7.0b; Windows NT 5.2; .NET CLR 1.1.4322; .NET CLR 2.0.50727; InfoPath.2; .NET CLR 3.0.04506.30)",
    "Mozilla/5.0 (Windows; U; Windows NT 5.1; zh-CN) AppleWebKit/523.15 (KHTML, like Gecko, Safari/419.3) Arora/0.3 (Change: 287 c9dfb30)",
    "Mozilla/5.0 (X11; U; Linux; en-US) AppleWebKit/527+ (KHTML, like Gecko, Safari/419.3) Arora/0.6",
    "Mozilla/5.0 (Windows; U; Windows NT 5.1; en-US; rv:1.8.1.2pre) Gecko/20070215 K-Ninja/2.1.1",
    "Mozilla/5.0 (Windows; U; Windows NT 5.1; zh-CN; rv:1.9) Gecko/20080705 Firefox/3.0 Kapiko/3.0",
    "Mozilla/5.0 (X11; Linux i686; U;) Gecko/20070322 Kazehakase/0.4.5",
    "Mozilla/5.0 (X11; U; Linux i686; en-US; rv:1.9.0.8) Gecko Fedora/1.9.0.8-1.fc10 Kazehakase/0.5.6",
    "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/535.11 (KHTML, like Gecko) Chrome/17.0.963.56 Safari/535.11",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_7_3) AppleWebKit/535.20 (KHTML, like Gecko) Chrome/19.0.1036.7 Safari/535.20",
    "Opera/9.80 (Macintosh; Intel Mac OS X 10.6.8; U; fr) Presto/2.9.168 Version/11.52",
]

headers = {
            'User-Agent': random.choice(USER_AGENTS),
        }


def spiderRequest(url, headers):
    try:
        response = requests.get(url, headers=headers)
        if response.status_code != 200:
            print("爬取异常！！！正在等待重新爬取。。。")
            time.sleep(5)
            return spiderRequest(url=url, headers=headers)
    except:
        print("爬取中断！！！正在等待重新爬取。。。")
        time.sleep(10)
        return spiderRequest(url=url, headers=headers)
    
    return response

first_url = 'https://bigbangtrans.wordpress.com/series-1-episode-1-pilot-episode/'
response = spiderRequest(first_url, headers)
# print(type(response))                 # 返回值的类型
# print(response.status_code)           # 当前网站返回的状态码
# print(type(response.status_code))     # int
# print(type(response.text))            # 网页内容的类型
# print(response.text)                  # 网页的具体内容（html代码）

# * 爬取目录链接
url_list = re.findall("href=\"(.*?)/\">Series", response.text)
print("共有{}个url待爬取".format(len(url_list)))

# # * 爬取台词
for idx, url in enumerate(url_list):
    print("正在爬取第{}个url...".format(idx+1))
    time.sleep(2)
    # response = requests.get(url, headers=headers)
    response = spiderRequest(url, headers)
    context = re.findall("Calibri;\">(.*?)</span></p>", response.text)
    if len(context) == 0:
        context = re.findall("<p>(.*?)</p>", response.text)
        if len(context) == 0:
            context = re.findall("sans-serif;\">(.*?)</span></p>", response.text)
    if len(context) == 0:
        print("异常！！！{}:{}".format(idx+1, url))
        
    with open('./spider/bigbang.txt', 'a') as f:
        for sample in context:
            f.write(sample+'\r\n')
            
print("爬取完成！")


