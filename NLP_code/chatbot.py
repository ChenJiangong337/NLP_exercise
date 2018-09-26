#   Filename:   chatbot.py

#   pynlpir应用

import pynlpir

pynlpir.open()

s = '聊天机器人到底该怎么做呢？'
segments = pynlpir.segment(s)
for segment in segments:
    print(segment[0],'\t',segment[1])

key_words = pynlpir.get_key_words(s,weighted=True)
for key_word in key_words:
    print(key_word[0],'\t',key_word[1])

s = '怎么才能把电脑里的垃圾文件删除'

key_words = pynlpir.get_key_words(s,weighted=True)
for key_word in key_words:
    print(key_word[0],'\t',key_word[1])

s = '海洋是如何形成的'
segments = pynlpir.segment(s,pos_names='all')
for segment in segments:
    print(segment[0],'\t',segment[1])

segments = pynlpir.segment(s,pos_names='all',pos_english=False)
for segment in segments:
    print(segment[0],'\t',segment[1])

pynlpir.close()

#   语言技术平台云LTP-Cloud
#   GET请求及返回结果示例：
#   curl -i -G --data-urlencode text="我是中国人。" -d "api_key=q2O0x753X2ONVSiz0sUPHU4QZS6Shb8TMs1wgk6t&pattern=dp&format=plain" https://api.ltp-cloud.com/analysis/
#   POST请求及返回结果示例：
#   curl -i --data-urlencode text="我是中国人。" -d "api_key=q2O0x753X2ONVSiz0sUPHU4QZS6Shb8TMs1wgk6t&pattern=dp&format=plain" "https://api.ltp-cloud.com/analysis/"
#   使用Python语言以POST方式调用REST API代码示例如下：

