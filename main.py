import os
import Modules
from Modules.NewsCrawler import NewsCrawler
from Modules.ClusteringModule import ClusteringModule
from Modules.SummaryModule import SummaryModule
from Modules.TwitterSearchModule import TwitterSearchModule

Crawler = NewsCrawler()
Model = SummaryModule()
TwitterSearch = TwitterSearchModule()

QueryWord = str(input("\nEnter a Query - "))

print("\n\nCrawling to find articles and relevant information...")

urls = list(Crawler.getEventRelevantURLs(QueryWord))
articlesDictionary = Crawler.getArticlesFromURLs(urls)
newsList = Crawler.cleanArticlesDict(articlesDictionary)
Crawler.saveDataToFile(newsList)

print("\nCrawl Complete\n\nIdentifyingKeywords...")

Model.printKeyWords(QueryWord)

Clusterer = ClusteringModule("../Files/embeddings.txt", Model.size) #Path relative to ClusteringModule script

newKeywords = Clusterer.reduceKeywords(ShowGraph=True)

print("\n\n", newKeywords)

print("\nSearching Twitter...")
TwitterSearch.searchTweets()


#code
# final_clusters=TwitterSearch.clusterCenter(2, 3, 3) 

# threashold for a connection in the hashtag graph we have
# minLengthOfComponent(min no of hashtags in cluster to consider it as a cluster )
# noCenters we want from each cluster

#code
# print(final_clusters)


# its a list of 2 array in each row first array all the cluster hashtags and second array in the same row is the selected hashtags 

# example Query word used is Johnny Depp
# """[[hashtag cluster 0-['#JohnnyDepp','#JusticeForJohnnyDepp','#AmberHeard','#AmberHeardIsALiar','#JohnnyDeppVsAmberHeardTrial','#AmberTurd','#JohnnyDeppvAmberHeard',
#    '#JohnnyDeppVsAmberHeard','#DeppvHeard','#DeppVsHeard','#DeppHeardTrial','#deppvsheardtrial','#AmberHeardlsAnAbuser','#WeJustDontLikeYouAmber',
#    '#IStandWithJohnnyDepp','#JusticeForJohnny','#AmberHeardlsALiar','#AmberHeardlsApsychopath','#bycottamberheard','#Aquaman2','#JohnnyDeppIsInnocent',
#    '#MeToo','#MenToo','#JohnnyDeppAmberHeardTrial','#JohnnyDepptrial','#AmberHeardIsAnAbuser','#AmberHeardDeservesPrison','#JusticeForJohhnyDepp','#amberheardisapsychopath',
#    '#AmberHeardIsAPsycopath','#AmberTurds','#WeLoveYouJohnnyDepp','#AbuserHeard','#AmberHeardIsAPsychopath','#AmberTurd💩',
#    '#EvaGreen','#EntertainmentNews','#PenelopeCruz','#ElonMusk','#DeppvsHeard','#alpacasforjohnny','#jhonnydepp','#camillevasquez','#support',
#    '#trial','#PiratesoftheCaribbean','#Disney','#JackSparrow','#CaptainJackSparrow','#megapint','#rum','#DomesticViolence','#GoodActor',
#    '#BringBackDepp','#Depp','#DefamationTrial','#Case','#Jury','#Stand','#Court','#Attorney','#Witness','#Testimony','#CamilleVasquez','#EllenBarkin','#piratesofthecarribean'],
#  selected centers ['#JohnnyDepp', '#AmberHeard', '#JusticeForJohnnyDepp']],
 
#  hashtag cluster 1 - [['#problem', '#attitude', '#leadership'],
#  selected centers ['#problem', '#attitude', '#leadership']],
 
# hashtag cluster 2- [['#johnnydepp', '#amberheard', '#justiceforjohnnydepp', '#captainjack'],
#   selected centers - ['#johnnydepp', '#justiceforjohnnydepp', '#amberheard']],
 
#  cluster 3 - [['#推特新号','#推特小号','#推特发帖','#推文代发','#购买推特','#出售推特','#twitter批发','#批发推特','#比特币','#以太坊','#推特引流','#推特粉丝','#点赞'],
#  selected centers- ['#推特新号', '#推特小号', '#推特发帖']],
 
#  cluster 4-[['#metoo','#metooinceste','#metooincest','#usa','#Indian','#chinese','#Brazil','#metooindia','#Indonesia','#Nigeria'],
#   selected centers-['#metoo', '#metooinceste', '#metooincest']],
 
#  cluster 5- [['#Johnny','#bitcoin','#Hilariously','#Perfect','#Fans','#Recreate','#Depps','#Captain','#Jack','#Sparrow','#Run'],
#   selected centers['#Johnny', '#bitcoin', '#Hilariously']],
 
#  cluster 6- [['#johnny', '#team', '#love'], 
#  selected centers - ['#johnny', '#team', '#love']]]"""

# TwitterSearchModule().wordCrawlingForOntology(final_clusters)
# this function is creating a dictionary of word that it found in news article, for each cluster found above




#code
# connection=TwitterSearch.Ontology()


# the connection the graph showing is number of connections between each cluster

#code
#final_hashtag_list=TwitterSearch.finalHashtagsssByDragonS(final_clusters,connection,1.0,80)



# 3rd parameter is ratio like the list of final selected clusters.. what ratio of hashtags we are taking from them
# The most connected hashtags from that cluster will be taken and the ratio we are specifying
# 4th parameter is the limit in the graph between cluster 0,1,2,3.... So here we are first selecting the cluster with  
# most number of hashtags and then checking the connection limit/weight with other clusters, whoever have more limit than the
# limit threashold will be selected as a cluster to give us r ratio of most connected hashtags from that cluster. 

# print(final_hashtag_list)





final_hashtag_list=TwitterSearch.old_ontology()
print(final_hashtag_list)
final_hashtag_list=TwitterSearch.new_ontology(newsList)
print(final_hashtag_list)


