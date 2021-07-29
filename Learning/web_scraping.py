#!/usr/bin/env python
# coding: utf-8

# Over time I've read/skimmed/seen interesting articles or article names from these people. While Medium does have an RSS feed, this only returns a page of the most recent articles. A historic TOC is not provided, you have to scroll. <br>
# <br>
# - This little project took 2 more days than it should have and only works for these two people.
# - Good first introduction to scraping, understanding how it all works which was something I wanted to get into for a while but didn't have the time. Good practice in python, writing functions and seeing how stuff works.
# <br>
# <br>
# 
# **Lessons learnt: Web scraping is a huge time sink and generally not worth it. Look hard for APIs before going down the scraping route:**
# - HTML is just fiddly. Shit changes too much, doesn't follow a consistent order. The medium stuff is all over the place.

# In[2]:


import requests
from bs4 import BeautifulSoup 
import pandas as pd
from selenium import webdriver
from selenium.webdriver.firefox.options import Options
import time
import re
from IPython.display import Audio
import numpy as np

from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"


# In[133]:


medium_bloggers = ['https://williamkoehrsen.medium.com/', 'https://actsusanli.medium.com/']
nn = ['wk','sli']


# In[125]:


def get_articles_from(n):
    global wk
#Webdriver--------------------------------    
    firefox_profile = webdriver.FirefoxProfile()
    firefox_profile.set_preference('permissions.default.image', 2)
    firefox_profile.set_preference('dom.ipc.plugins.enabled.libflashplayer.so', 'False')          
    
    options = Options()
    options.headless = True
    driver = webdriver.Firefox(options=options,firefox_profile=firefox_profile)
    #driver = webdriver.Firefox()
        
    driver.get(medium_bloggers[n])
    time.sleep(6)

    t = 10 
    while True:
        try:
            p = "/html/body/div/div/div[3]/div[2]/div/div[2]/div/div[" + str(t) + "]/div[1]/button"
            loadMoreButton = driver.find_element_by_xpath(p).click()
            #time.sleep(1)
            #loadMoreButton.click()
            time.sleep(5)
            t+= 10

        except Exception as e:
            #print(e)
            print(str(int(t/10)) + ' pages loaded')       
            break
    
    da_sauce = driver.page_source
    driver.quit()

    
    #BeautifulSoup--------------------------------    
    soup = BeautifulSoup(da_sauce)
    #Titles
    soup_titles = soup.find_all('a',"dm bq")
    titles = [soup_titles[i].get_text() for i in range(len(soup_titles))]
    
    soup_urls = [a['href'] for a in soup.find_all("a","dm bq",href=True)]
    bloggers_links = [medium_bloggers[n][:len(medium_bloggers[n])-1] if medium_bloggers[n].endswith('/') else i for i in range(len(medium_bloggers))]
    urls = [(bloggers_links[n]+ l) if l.startswith('/') else l for l in soup_urls] 



    #Blurb
    soup_blurb = soup.find_all('h2',"hw de he au b hx hy hz ia ib ic id ie if ig ih ii ij bs")   


    if not soup_blurb:

        soup_sections = soup.find_all('section','dg gc gd db ge')
        blurb= []
        for i in range(len(titles)):
            if soup_sections[i].find('h2', "hv de hd au b hw hx hy hz ia ib ic id ie if ig ih ii bs") is not None:
                s = soup_sections[i].find('h2', "hv de hd au b hw hx hy hz ia ib ic id ie if ig ih ii bs").get_text()
            else: 
                if soup_sections[i].find('strong', "il ct") is not None:
                    s = soup_sections[i].find('strong', "il ct").get_text()            
                else: 
                    if soup_sections[i].find('p','ij ik hd il b hz hf im ib hi in io ip iq ir is it iu iv iw pe iy dg fy') is not None:
                        s = soup_sections[i].find('p','ij ik hd il b hz hf im ib hi in io ip iq ir is it iu iv iw pe iy dg fy').get_text()
                    else: s = 'None'                      

            blurb.append(s)


        blurb= [i for i in blurb if i!= 'None']

        blurb.extend(['None' for i in range(len(titles)-len(blurb))])


    if n==1:


        r = re.compile(r"\bwikipedia\b")
        wiki_delete = [i for i, s in enumerate(urls, start=0) if re.search(r, s)]
        for i in sorted(wiki_delete, reverse=True):
            del urls[i],
            del titles[i]
        blurb = [soup_blurb[i].get_text() for i in range(len(soup_blurb))]
        blurb.extend(['None' for i in range(len(titles)-len(blurb))])
                  

    

    def make_clickable(val):
        return '<a target="_blank" href="{}">{}</a>'.format(val, val)

    
    wk = pd.DataFrame({'titles':titles,'blurbs':blurb,'urls':urls})
    wk.to_csv('/home/r/Documents/python_linux/articles_from_' + str(nn[n]) +'.csv')
    wk = wk.style.set_properties(**{'text-align': 'right'})              .format({'urls': make_clickable})
    
    return wk


# In[134]:


get_articles_from(1)
wave = np.sin(8*np.pi*500*np.arange(10000*0.15)/10000)
Audio(wave, rate=10000, autoplay=True)


# In[127]:


wk


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




