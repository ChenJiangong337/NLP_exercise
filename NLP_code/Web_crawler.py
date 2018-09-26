# Filename： Web crawler

from urllib.request import urlopen
from bs4 import BeautifulSoup
from selenium import webdriver
import csv

#  （1）：网易云音乐歌单

html = urlopen('http://jr.jd.com')
print(html.read())
html.close()

html = urlopen('http://jr.jd.com')
bs_obj = BeautifulSoup(html.read(),'html.parser')
text_list = bs_obj.find_all('a','nav-item-primary')
for text in text_list:
    print(text.get_text())
html.close()

url = 'http://music.163.com/#/discover/playlist/'\
      '?order=hot&cat=%E5%85%A8%E9%83%A8&limit=35&offset=0'

driver = webdriver.PhantomJS(executable_path='/Users/chenjiangong/Documents/phantomjs-2.1.1-macosx/bin/phantomjs')

csv_file = open('playlist.csv','w',newline='')
writer = csv.writer(csv_file)
writer.writerow(['title','playtimes','link'])

while url != 'javascript:void(0)':
    driver.get(url)
    driver.switch_to.frame('contentFrame')
    data = driver.find_element_by_id('m-pl-container').\
           find_elements_by_tag_name('li')
    for i in range(len(data)):
        nb = data[i].find_element_by_class_name('nb').text
        if '万' in nb and int(nb.split('万')[0])>500:
            msk = data[i].find_element_by_css_selector('a.msk')
            writer.writerow([msk.get_attribute('title'),
                             nb,msk.get_attribute('href')])
    url = driver.find_element_by_css_selector('a.zbtn.znxt').\
          get_attribute('href')
csv_file.close()

#   （2）：迷你爬虫架构

#cur_depth = 0
#depth = conf.max_depth
#while cur_depth <= depth:
#    for host in hosts:
#        url_quene.put(host)
#        time.sleep(conf.crawl_interval)
#    cur_depth += 1
#    web_parse.cur_depth = cur_depth
#    url_queue.join()
#    hosts = copy.deepcopy(u_table.todo_list)
#    u_table.todo_list = []
