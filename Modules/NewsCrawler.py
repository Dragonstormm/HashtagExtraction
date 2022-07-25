import os

from newsplease import NewsPlease
import requests
from bs4 import BeautifulSoup
import re
import json

class NewsCrawler:

    def __init__(self):
        self.contentThreshhold = 220
        self.numberOfLinksRequired = 50
        pass
    
    """
    Step 1: Get URLs related to a user specified event
    Returns a set of URLs of various websites containing information/articles related to the event
    """ 
    def getEventRelevantURLs(self,eventQuery):
        print("Event Query: " + eventQuery)
        articleNumber = 0
        urls=set()
        while articleNumber < self.numberOfLinksRequired: 
            page = requests.get("https://www.google.com/search?q="+eventQuery + "&tbm=nws&start=" + str(articleNumber))
            soup = BeautifulSoup(page.content,features="lxml")
            # print(soup)
            
            for link in  soup.find_all("a",href=re.compile("(?<=/url\?q=)(htt.*://.*)")):
                urlWithAmpersand = re.split(":(?=http)",link["href"].replace("/url?q=",""))[0]
                finalURL = urlWithAmpersand.split('&')[0]
                # Handling some unwanted google links (support.google.com , accounts.google.com)
                if(finalURL.__contains__("google.com")):
                    continue
                else:
                    urls.add(finalURL)
            articleNumber+=10
        return urls
    
    """
    Step 2: Scrape these URLs for relevant information
    Returns a dictionary of article JSON objects pertaining to each URL
    """ 
    def getArticlesFromURLs(self,eventURLs):
        articlesDict=dict()
        for url in eventURLs[0:len(eventURLs)-2]:
            try:
                articlesDict[url] = NewsPlease.from_url(url,timeout=60)
            except:
                pass
        return articlesDict

    """
    Step 3: Cleaning the input, by appling some filters:
        a) Removing content which is of less than a threshhold length, removing certain errors or strings not parsed correctly etc.
        Returns a list of articles
    """ 
    def cleanArticlesDict(self,articlesDict):
        newsContent = []
        for url in articlesDict:
            try:

                if(len(articlesDict[url].maintext)<=self.contentThreshhold):
                    pass
                else:
                    x = articlesDict[url].maintext
                    newsContent.append(x)
            except:
                pass
        return newsContent

    def saveDataToFile(self,data):
        dirname = os.path.dirname(__file__)
        path = os.path.join(dirname,"../Files/newsArticles.json")
        file = open(path, "w")
        json.dump(data, file)


#print("Starting")
#myCrawler = NewsCrawler()
#urls = myCrawler.getEventRelevantURLs("bollywood")
#articlesDictionary = myCrawler.getArticlesFromURLs(urls)
#newsList = myCrawler.cleanArticlesDict(articlesDictionary)
#myCrawler.saveDataToFile(newsList)
#myCrawler.loadDataFromFile()
#print("Ending")