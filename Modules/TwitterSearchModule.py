from math import floor
from sklearn.metrics.pairwise import cosine_similarity
import tweepy
import os
import json
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
from .NewsCrawler import NewsCrawler
import warnings
API_KEY = "AAAAAAAAAAAAAAAAAAAAAOGxYQEAAAAAUS1JGXfkVIMn1d8aaCay4nxFmxk%3DtE4CGvqr3CkOVXwutGmB6VWJTjKmxvs0slchJjrrL9P9G17NmF"

class Hashtag:
    def __init__(self,hashtagName,listOfTweets,frequency):
        self.hashtagName= hashtagName
        self.listOfTweets=listOfTweets
        self.frequency = frequency
    
    

class TwitterSearchModule:
    def __init__(self):
        self.client = tweepy.Client(API_KEY)
        self.hashtagObjects=[]
        self.adjacencyMatrix=[]
        self.keywordEmbeddings=[]
        self.seedEmbeddings=[]
        self.tweetsPerKeyword={}
        self.maxNumberOfTweets=100
        self.minNumberOfTweets=20
        self.keywordList=[]

    def loadClusteredKeywords(self):
        dirname = os.path.dirname(__file__)
        path = os.path.join(dirname,"../Files/clusteredKeywords.json")
        file = open(path, "r")
        data = json.load(file)
        # print(data)
        return data
        
        
    def assignDynamicTweets(self,keywordList):
        self.loadKeywordEmbeddings()
        self.loadSeedEmbeddings()
        similarityList=[]
        # Extracting the embedding of each keyword from the saved map, and then using it to compare with embedding of the seed word
        for keyword in keywordList:
            currentKeywordEmbedding = np.array(self.keywordEmbeddings[keyword])
            similarity = cosine_similarity(currentKeywordEmbedding.reshape(1,-1),np.array(self.seedEmbeddings).reshape(1,-1))
            similarityList.append([keyword,similarity])
        # similarityList sorted by its similarity -> [["hello i am keyword",0.5123],.....]
        similarityList.sort(reverse=True, key=lambda x : x[1])
        maxSimilarity = similarityList[0][1]
        minSimilarity = similarityList[len(similarityList)-1][1]
        # Basic min-max normalization for deciding number of tweets
        for i in range(0,len(similarityList)):
            currentSimilarity = similarityList[i][1]
            normalizedNumber = (currentSimilarity-minSimilarity)/(maxSimilarity-minSimilarity)
            normalizedNumber = floor(normalizedNumber*(self.maxNumberOfTweets-self.minNumberOfTweets)) + self.minNumberOfTweets
            self.tweetsPerKeyword[similarityList[i][0]] = normalizedNumber
        print(self.tweetsPerKeyword)
        pass

    def searchTweets(self):
        tweets=[]
        queryParameters = " lang:en -is:retweet"    #SPACE IN START IS NECESSARY, - negates the retweet attribute, language is set to be English
        self.keywordList = self.loadClusteredKeywords()
        self.assignDynamicTweets(self.keywordList)
        # tweetPerKeyword = 10
        for keyword in self.keywordList:
            for tweet in tweepy.Paginator(self.client.search_recent_tweets, keyword + queryParameters, max_results=self.tweetsPerKeyword[keyword],tweet_fields=["text"]).flatten(self.tweetsPerKeyword[keyword]):
                tweets.append(tweet.text)
        # print(tweets)
        #hashtags = 
        self.extractHashtags(tweets)
        self.saveTweetsToFile(tweets)
        # print(hashtags)
        self.hashtagObjects.sort(key=lambda x: x.frequency, reverse=True)
        # self.printHashtags()
        # sortHashtagsByFrequency = sorted(self.hashtagObjects.items(), key=lambda x: x[1].frequency, reverse=True)
        self.saveHashtags()
        self.createGraph()
        print(len(tweets))
        # centerss=self.clusterCenter(3,3,3) # minimum edge weight condition in matrix and minimum number of nodes present in the connected component to be considered as a cluster and number of cluster centers we want
        # print(centerss)
        # self.Ontology()
    # def searchTweetsSecondTime(self):
        
    def extractHashtags(self,tweets):
        hashtagsDict=dict() #Access hashtag Object by hashtag name
        hashtagEndings = [' ','\n','|','#','\\u',',','.']
        tweetIndex = 0
        # print(tweets)
        for tweet in tweets:
            try:
                for i in range(0,len(tweet)):
                    # print(tweet)
                    if(tweet[i]=='#'):
                        hashtagEnding = min(tweet.find(char,i+1) for char in hashtagEndings if char in tweet[i+1:])
                        # print(hashtagEnding)
                        hashtag = tweet[i:hashtagEnding]
                        
                        if hashtag in hashtagsDict:
                            temporaryObject = hashtagsDict[hashtag]
                            temporaryObject.frequency = temporaryObject.frequency + 1
                            temporaryObject.listOfTweets.append(tweetIndex)
                            # print(hashtag +"inside if")
                            # print(temporaryObject.listOfTweets)
                        else:
                            hashtagObject = Hashtag(hashtagName=hashtag,listOfTweets=[],frequency=0)
                            hashtagObject.frequency = 1
                            hashtagObject.listOfTweets.append(tweetIndex)
                            self.hashtagObjects.append(hashtagObject)
                            # print(hashtag +  "inside else")
                            # print(hashtagObject.listOfTweets)
                            hashtagsDict[hashtag] = hashtagObject
                            # hashtagObject = None
                        i = hashtagEnding
                        # print(hashtag + "$$$$$$$$$$$$$$$")
                        # print(hashtagsDict[hashtag])
                        
            except Exception as e:
                print(e)
                pass
            tweetIndex = tweetIndex + 1
        # print(hashtagsDict)
        #return hashtagsDict


    def saveTweetsToFile(self,tweets):
        dirname = os.path.dirname(__file__)
        path = os.path.join(dirname,"../Files/tweets.txt")
        with open(path, "w",encoding='UTF-8') as outfile:
            outfile.write("\n\n\nNEXT TWEET\n\n\n".join(str(item) for item in tweets))
    
    def saveHashtags(self):
        dirname = os.path.dirname(__file__)
        path = os.path.join(dirname,"../Files/hashtagsTEMP.txt")
        hashtagJson=[]
        for hashtagObject in self.hashtagObjects:
            tempJSON = {
                "hashtagName": hashtagObject.hashtagName,
                "frequency": hashtagObject.frequency,
                "listOfTweets":hashtagObject.listOfTweets
            }
            hashtagJson.append(tempJSON)
        with open(path, 'w') as convert_file:
            convert_file.write(json.dumps(hashtagJson))
    
    def loadHashtagObjects(self):
        dirname = os.path.dirname(__file__)
        path = os.path.join(dirname,"../Files/hashtagsTEMP.txt")  
        file = open(path, "r")
        data = json.load(file)
        self.hashtagObjects = []
        #print(data)
        for item in data:
            self.hashtagObjects.append(Hashtag(item['hashtagName'],item['listOfTweets'],item['frequency']))
        #self.printHashtags()

    def loadKeywordEmbeddings(self):
        dirname = os.path.dirname(__file__)
        path = os.path.join(dirname,"../Files/keywordsAndEmbeddings.json")  
        file = open(path, "r")
        data = json.load(file)
        self.keywordEmbeddings = data
        pass

    def loadSeedEmbeddings(self):
        dirname = os.path.dirname(__file__)
        path = os.path.join(dirname,"../Files/seedEmbeddings.json")  
        file = open(path, "r")
        data = json.load(file)
        self.seedEmbeddings = data[0]   #as list [seedEmbeddings] was stored
        # print(self.seedEmbeddings)
        # print("#################")
        pass

    def printHashtags(self):
        print("inside printing")
        for hashtag in self.hashtagObjects:
            print(hashtag.hashtagName + ": " + str(hashtag.frequency))
            print(hashtag.listOfTweets)

    def show_graph(self):
        rows, cols = np.where(np.array(self.adjacencyMatrix) >= 4)
        edges = zip(rows.tolist(), cols.tolist())
        weights = [self.adjacencyMatrix[e[0]][e[1]] for e in edges]
        edges = zip(rows.tolist(), cols.tolist(), weights)
        gr = nx.Graph()
        gr.add_weighted_edges_from(edges, weight='weight')
        labels={}
        # print(rows)
        for i in range(0, len(self.hashtagObjects)):
            if i in rows:
                labels[i] = self.hashtagObjects[i].hashtagName
            if i in cols:
                labels[i] = self.hashtagObjects[i].hashtagName
        #print(labels)
        pos=nx.spring_layout(gr)
        weightLabels = nx.get_edge_attributes(gr,'weight')
        nx.draw(gr, pos=pos, node_size=500,labels= labels,font_size=8, with_labels=True)
        nx.draw_networkx_labels(gr,  pos, labels = labels, font_size=8)
        nx.draw_networkx_edge_labels(gr, pos, edge_labels= weightLabels,font_size=8)
        plt.show()

    def createGraph(self):
        self.loadHashtagObjects()
        
        for i in range(0,len(self.hashtagObjects)):
            self.adjacencyMatrix.append([])
            for j in range(0,len(self.hashtagObjects)):
                if i==j:
                    self.adjacencyMatrix[i].append(0)
                    continue
                list1 = self.hashtagObjects[i].listOfTweets
                list2 = self.hashtagObjects[j].listOfTweets
                intersection = [value for value in list1 if value in list2]
                self.adjacencyMatrix[i].append(len(intersection))
        #print(self.adjacencyMatrix)
        
        # for i in range(0,len(self.adjacencyMatrix)):
        #     for j in range(0,len(self.adjacencyMatrix[i])):
        #         print(self.adjacencyMatrix,sep=" ")
        #     print("")
        
        print("HashtagObject Length - ", len(self.hashtagObjects))
        self.show_graph()
    
    def saveClusterCenters(self,final_clusters):
        dirname = os.path.dirname(__file__)
        path = os.path.join(dirname,"../Files/ClusterCenters.txt")
        with open(path, "w") as f:
            f.write(json.dumps(final_clusters))

    def openClusterCenters(self):
        dirname = os.path.dirname(__file__)
        path = os.path.join(dirname,"../Files/ClusterCenters.txt")
        with open(path, 'r') as f:
            data = json.loads(f.read())
        return data
    def saveClusterConnection(self,connection):
        dirname = os.path.dirname(__file__)
        path = os.path.join(dirname,"../Files/ClusterConnection.txt")
        np.savetxt(path, connection)

    def openClusterConnection(self):
        dirname = os.path.dirname(__file__)
        path = os.path.join(dirname,"../Files/ClusterConnection.txt")
        data=np.loadtxt(path, dtype=int)
        return data

    def DFSUtil(self, ConnectedComponent, matrix, v, visited, hashtags):
        visited[v] = True
        ConnectedComponent.append(hashtags[v])
        for i in range(len(matrix[v])):
            if visited[i] == False and matrix[v][i]>0:
                ConnectedComponent = self.DFSUtil(ConnectedComponent, matrix, i, visited, hashtags)
        return ConnectedComponent

    def connectedComponents(self, matrix, hashtags, minLengthOfComponent):
        visited = []
        cc = []
        #print("MATRIX - ", matrix)
        for i in range(len(matrix[0])):
            visited.append(False)
        for i in range(len(matrix[0])):
            if visited[i] == False:
                ConnectedComponent = []
                temp_component=self.DFSUtil(ConnectedComponent, matrix, i, visited, hashtags)
                if(len(temp_component)>=minLengthOfComponent):
                    cc.append(temp_component)
        return cc

    def clusterCenter(self, threashold=1, minLengthOfComponent=2, noCenters=2):
        hashtags=[]
        for i in range(len(self.hashtagObjects)):
            hashtags.append(self.hashtagObjects[i].hashtagName)
        matrix=self.adjacencyMatrix
        #print(self.adjacencyMatrix)
        #print(hashtags)
        #print("MATRIX CREATION - ", matrix)
        hashtags=np.array(hashtags)
        matrix=np.array(matrix)

    #     Converting the matrix in the values above the threshold 
        for i in range(len(matrix)):
            for j in range(len(matrix[0])):
                if matrix[i][j]<threashold:
                    matrix[i][j]=0
        
        hashtagIndex={}
        for i in range(len(self.hashtagObjects)):
            hashtagIndex[self.hashtagObjects[i].hashtagName]=i
        #print("MATRIX1 - ", matrix)
        cc=self.connectedComponents(matrix, hashtags, minLengthOfComponent)
        cc=np.array(cc)
        final_clusters=[]
        for i in range(len(cc)):
            index=[]
            for tag in cc[i]:
                index.append([tag,matrix[hashtagIndex[tag]].sum()])
            index=sorted(index, key=lambda x: x[1],reverse=True)
            selected=[]
            noob=[]
            for j in range(min(noCenters,len(index))):
                selected.append(index[j][0])
            for j in range(len(index)):
                noob.append(index[j][0])
            final_clusters.append([noob,selected])
        self.saveClusterCenters(final_clusters)
        return final_clusters

    def split(self,txt, seps):
        default_sep = seps[0]

        # we skip seps[0] because that's the default separator
        for sep in seps[1:]:
            txt = txt.replace(sep, default_sep)
        return [i.strip() for i in txt.split(default_sep)]

    def wordCrawlingForOntology(self,final_clusters):
        occurance=[]
        for i in range(len(final_clusters)):
            wordDict={}
            for word in final_clusters[i][1]:
                myCrawler = NewsCrawler()
                urls = myCrawler.getEventRelevantURLs(word[1:])
                articlesDictionary = myCrawler.getArticlesFromURLs(list(urls))
                newsList = myCrawler.cleanArticlesDict(articlesDictionary)
                for article in newsList:
                    wordList=words=self.split(article,(' ',',','(',')','<','>','?','-',':','.','"',';','/','*','+','_','!','=','#','$','%','^','&','|','{','}','[',']','@','\'','\\','~','`'))   
                    
                    posTaggedResult = nltk.tag.pos_tag(wordList)
                    stp=list()
                    for w in posTaggedResult:
                        if(w[1]!="NN" and w[1]!="NNP"):
                            stp.append(w[0])
                    
                    for key in wordList:
                        # Going through each word and adding it in dictionary if not present else increasing its frequency
                        if(len(key)==0 or key in stopwords.words('english') or len(key)==1 or key in stp):
                            continue
                        if key in wordDict:
                            wordDict[key]+=1
                        else:
                            wordDict[key]=1
            occurance.append(wordDict)
        return occurance

    def plotOntologyRelation(self,x):
        rows, cols = np.where(np.array(x))
        edges = zip(rows.tolist(), cols.tolist())
        weights = [x[e[0]][e[1]] for e in edges]
        edges = zip(rows.tolist(), cols.tolist(), weights)
        gr = nx.Graph()
        gr.add_weighted_edges_from(edges, weight='weight')
        labels={}
        # print(rows)
        for i in range(0, len(x)):
            if i in rows:
                labels[i] = i
            if i in cols:
                labels[i] = i
        #print(labels)
        pos=nx.spring_layout(gr)
        weightLabels = nx.get_edge_attributes(gr,'weight')
        nx.draw(gr, pos=pos, node_size=500,labels= labels,font_size=8, with_labels=True)
        nx.draw_networkx_labels(gr,  pos, labels = labels, font_size=8)
        nx.draw_networkx_edge_labels(gr, pos, edge_labels= weightLabels,font_size=8)
        plt.show()

    def Ontology(self):
        final_clusters=self.openClusterCenters()
        occurance=self.wordCrawlingForOntology(final_clusters)
        connection=np.zeros((len(final_clusters),len(final_clusters)))
        for i in range(len(occurance)):
            for j in range(len(occurance)):
                if(i==j):
                    continue
                count=0
                for word in final_clusters[j][1]:
                    word=word[1:]
                    if(word in occurance[i]):
                        count+=occurance[i][word]
                connection[i][j]=connection[i][j]+count
                connection[j][i]=connection[j][i]+count
        #print("BEFORE GRAPH")
        self.plotOntologyRelation(connection)
        #print("AFTER GRAPH")
        self.saveClusterConnection(connection)
        #print("AFTER SAVE")
        return connection

    def finalHashtagsssByDragonS(self,final_clusters,connection,ratio=0.4,limit=100):
        max_len=0
        index_of_biggest_cluster=0
        index=0
        for line in final_clusters:
            if(len(line[0])>max_len):
                max_len=len(line[0])
                index_of_biggest_cluster=index
            index+=1
        
        index_list_of_final_clusters=[]
        index_list_of_final_clusters.append(index_of_biggest_cluster)
        for j in range(0,len(connection[index_of_biggest_cluster])):
            if(connection[index_of_biggest_cluster][j]>=limit):
                index_list_of_final_clusters.append(j)
        # print(index_list_of_final_clusters)
        final_hashtag_list=[]
        for indece in index_list_of_final_clusters:
            lenn=len(final_clusters[indece][0])
            for j in range(0,(int)(ratio*lenn)):
                final_hashtag_list.append(final_clusters[indece][0][j])
        # print(final_hashtag_list)
        return final_hashtag_list


    def seed_crawl_dict(self,newsList):
        wordDict={}
        for article in newsList:
            wordList=words=self.split(article,(' ',',','(',')','<','>','?','-',':','.','"',';','/','*','+','_','!','=','#','$','%','^','&','|','{','}','[',']','@','\'','\\','~','`'))   
            posTaggedResult = nltk.tag.pos_tag(wordList)
            stp=list()
            for w in posTaggedResult:
                if(w[1]!="NN" and w[1]!="NNP"):
                    stp.append(w[0])
        
            for key in wordList:
                # Going through each word and adding it in dictionary if not present else increasing its frequency
                if(len(key)==0 or key in stopwords.words('english') or len(key)==1 or key in stp):
                    continue
                if key in wordDict:
                    wordDict[key]+=1
                else:
                    wordDict[key]=1
        return wordDict
    
    def Ontology2(self,occurance,seed_dict):
        connection=np.zeros((len(occurance)))
        index=0
        seed_dict=dict(seed_dict)
        for w,value in sorted(seed_dict.items(), key=lambda kv: kv[1], reverse=True):
    #         print(w,value)
            if(index>20):
                break
            for i in range(len(occurance)):
                if(w in occurance[i]):
                    connection[i]+=min(value,occurance[i][w])
            index+=1
        return connection
    
    def finalHashtagsssByDStorm(self,final_clusters,connection,ratio=0.4,limit=400):
        index_list_of_final_clusters=list()
        maximum_connection_value=0
        for i in range(len(connection)):
            maximum_connection_value=max(maximum_connection_value,connection[i])
            if(connection[i]>=limit):
                index_list_of_final_clusters.append(i)

        # print(index_list_of_final_clusters)
        final_hashtag_list=[]
        for indece in index_list_of_final_clusters:
            lenn=len(final_clusters[indece][0])
            artificial_ratio=connection[indece]/maximum_connection_value

    
            for j in range(0,(int)(ratio*lenn*artificial_ratio)):
                final_hashtag_list.append(final_clusters[indece][0][j])
        # print(final_hashtag_list)
        return final_hashtag_list

    
    def old_ontology(self):
        final_clustersss=self.clusterCenter(3, 3, 3)
        connectionnn=self.Ontology()
        final_hashtag_list=self.finalHashtagsssByDragonS(final_clustersss,connectionnn,1.0,80)
        return set(final_hashtag_list)
    
    
    def new_ontology(self,newsList):
        final_clustersss=self.clusterCenter(3, 3, 3)
        occuurancee=self.wordCrawlingForOntology(final_clustersss)
        seed_dict=self.seed_crawl_dict(newsList)
        connectionnn=self.Ontology2(occuurancee,seed_dict)
        
        # For graph
        arr=np.zeros((len(connectionnn)+1,len(connectionnn)+1))
        for i in range(len(connectionnn)):
            arr[0][i+1]=connectionnn[i]
        self.plotOntologyRelation(arr)
        # graph end

        final_hashtag_list=self.finalHashtagsssByDStorm(final_clustersss,connectionnn,0.5,400)
        return set(final_hashtag_list)
















# TwitterSearchModule().extractHashtags(["#omicron #india #covid #kerala #deaths #government #delhi #capital #maharashtra https://t.co/673ybhkZr",
#                                     "#omicron  #capital #maharashtra https://t.co/673ybhkZr",
#                                     "#omicron #india  https://t.co/673ybhkZr"      
#                                                         ])
# TwitterSearchModule().searchTweets(["Ukraine","Crisis","War"])   # if want to search fresh tweets
# TwitterSearchModule().loadHashtagObjects()
# TwitterSearch = TwitterSearchModule()
# TwitterSearch.createGraph()   # If want to access using saved file
# print(TwitterSearch.adjacencyMatrix)
# Calling of the cluster center function just give the threashold(same value given at the time of creating graph the edge weight condition)
# minLengthOfComponent(the minimum number of nodes present in the connected component to be considered as a cluster of out use)
# noCenters(number of cluster centers we want)
# final_clusters=TwitterSearch.clusterCenter(4, 3, 3)