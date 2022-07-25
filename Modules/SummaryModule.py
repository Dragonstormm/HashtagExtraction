import os

from summarizer import Summarizer
from keybert import KeyBERT
import json
import nltk
from nltk.corpus import stopwords
import warnings

#KeyBERT
import numpy as np
from tqdm import tqdm
from typing import List, Union, Tuple
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

# KeyBERT
from keybert._mmr import mmr
from keybert._maxsum import max_sum_similarity
from keybert._highlight import highlight_document
from keybert.backend._utils import select_backend

class KeywordModule(KeyBERT):
        def __init__(self, model):
                KeyBERT.__init__(self, model)

        def extract_keywords(self,
                              doc: str,
                              candidates: List[str] = None,
                              keyphrase_ngram_range: Tuple[int, int] = (1, 1),
                              stop_words: Union[str, List[str]] = 'english',
                              top_n: int = 5,
                              use_maxsum: bool = False,
                              use_mmr: bool = False,
                              diversity: float = 0.5,
                              nr_candidates: int = 20,
                              vectorizer: CountVectorizer = None,
                              seed_keywords: List[str] = None) -> List[Tuple[str, float]]:
                """ Extract keywords/keyphrases for a single document

                Arguments:
                doc: The document for which to extract keywords/keyphrases
                candidates: Candidate keywords/keyphrases to use instead of extracting them from the document(s)
                keyphrase_ngram_range: Length, in words, of the extracted keywords/keyphrases
                stop_words: Stopwords to remove from the document
                top_n: Return the top n keywords/keyphrases
                use_mmr: Whether to use Max Sum Similarity
                use_mmr: Whether to use MMR
                diversity: The diversity of results between 0 and 1 if use_mmr is True
                nr_candidates: The number of candidates to consider if use_maxsum is set to True
                vectorizer: Pass in your own CountVectorizer from scikit-learn
                seed_keywords: Seed keywords that may guide the extraction of keywords by
                                steering the similarities towards the seeded keywords

                Returns:
                keywords: the top n keywords for a document with their respective distances
                        to the input document
                """
                try:
                        # Extract Words
                        if candidates is None:
                                if vectorizer:
                                        count = vectorizer.fit([doc])
                                else:
                                        count = CountVectorizer(ngram_range=keyphrase_ngram_range, stop_words=stop_words).fit([doc])
                                        candidates = count.get_feature_names()

                        # Extract Embeddings
                        doc_embedding = self.model.embed([doc])
                        candidate_embeddings = self.model.embed(candidates)

                        # Guided KeyBERT with seed keywords
                        if seed_keywords is not None:
                                seed_embeddings = self.model.embed([" ".join(seed_keywords)])
                                doc_embedding = np.average([doc_embedding, seed_embeddings], axis=0, weights=[3, 1])

                        # Calculate distances and extract keywords
                        if use_mmr:
                                keywords = mmr(doc_embedding, candidate_embeddings, candidates, top_n, diversity)
                        elif use_maxsum:
                                keywords = max_sum_similarity(doc_embedding, candidate_embeddings, candidates, top_n, nr_candidates)
                        else:
                                distances = cosine_similarity(doc_embedding, candidate_embeddings)
                                keywords = [(candidates[index], round(float(distances[0][index]), 4))
                                        for index in distances.argsort()[0][-top_n:]][::-1]
                                self.embeddings = [(candidate_embeddings[index]) for index in distances.argsort()[0][-top_n:]][::-1]

                        return keywords,seed_embeddings
                except ValueError:
                        return []

class SummaryModule:
        def __init__ (self):
                self.model = Summarizer()     #bert-extractive summarizer
                self.KeyWordModel = KeywordModule('multi-qa-MiniLM-L6-cos-v1')    #keybert
                self.keywords = []      #list of generated keywords
                self.duplicates = []
                #nltk tags to identify each word in a sentence, removing words like adjectives, verbs, adverbs, e.t.c
                self.StopWordTags = ['NN', 'NNS', 'NNP', 'NNPS']  #not stop words
                self.StopWords = []
                self.embeddings = []
                #loading resources for nltk functions
                try:
                        nltk.tag.pos_tag("Test Sentence 1".split())
                        stopwords.words('english')
                except:
                        nltk.download('averaged_perceptron_tagger')
                        nltk.download('stopwords')
                #supressing nltk tokensiation warnings
                warnings.filterwarnings("ignore", category=UserWarning)
              
        def saveEmbeddingsToFile(self,data):
                #file = open("embeddings.json", "w")
                #json.dump(data, file)
                dirname = os.path.dirname(__file__)
                path = os.path.join(dirname,"../Files/embeddings.txt")
                np.savetxt(path, data)
                
        def saveKeywordsToFileJSON(self,data):
                dirname = os.path.dirname(__file__)
                path = os.path.join(dirname,"../Files/keywords.json")
                file = open(path, "w")
                json.dump(data, file)
        
        def saveKeywordsSortedToFile(self,data):
                dirname = os.path.dirname(__file__)
                path = os.path.join(dirname,"../Files/keywordsSorted.txt")
                with open(path, "w") as outfile:
                        outfile.write("\n".join(str(item) for item in data))
        
        def saveKeywordsWithEmbeddings(self,keywords,embeddings):       #Dictionary of keywords with its embeddings
                dirname = os.path.dirname(__file__)
                path = os.path.join(dirname,"../Files/keywordsAndEmbeddings.json")
                file = open(path, "w")
                data={}
                for i in range(0,len(keywords)):
                        data[keywords[i][0]] = embeddings[i].tolist()
                # print(data)
                json.dump(data, file)

        def saveSeedWordEmbeddings(self,seedEmbeddings):
                dirname = os.path.dirname(__file__)
                path = os.path.join(dirname,"../Files/seedEmbeddings.json")
                file = open(path, "w")
                data= seedEmbeddings.tolist()
                json.dump(data,file)

        def loadDataFromFile(self):    #load articles aquired from crawling
                dirname = os.path.dirname(__file__)
                path = os.path.join(dirname,"../Files/newsArticles.json")
                file = open(path, "r")
                data = json.load(file)
                return data

        def generateKeyWords(self, QueryWord):     
                articlesList = self.loadDataFromFile()       #get all articles
                
                for article in articlesList:                                                            #iteratting over each article
                        inputText = article
                        result = self.model(inputText, min_length = 90, ratio = 0.25)              #summarizing article
                        posTaggedResult = nltk.tag.pos_tag(result.split())
                        #identifying words like adjectives, verbs, e.t.c, and marking them as stop words
                        for word in posTaggedResult:
                                if (word[1] not in self.StopWordTags):
                                        self.StopWords.append(word[0])
                        self.StopWords.extend(stopwords.words('english')) #adding general list of english stop words
                        keywordList, seedEmbeddings = self.KeyWordModel.extract_keywords(result, stop_words=self.StopWords, seed_keywords=[QueryWord], keyphrase_ngram_range= (1,4))       #aquiring keywords
                        for keyword in keywordList:                                                     #appending to keyword list while ignoring duplicates
                                if (keyword[0] not in self.duplicates):
                                        self.keywords.append(keyword)
                                        self.duplicates.append(keyword[0])
                                        self.embeddings.append(self.KeyWordModel.embeddings[keywordList.index(keyword)])
                                else:
                                        pass
                        # print(seedEmbeddings)
                        # print(self.embeddings[0])
                self.saveEmbeddingsToFile(self.embeddings)
                self.saveKeywordsToFileJSON(self.keywords) 
                self.saveKeywordsWithEmbeddings(self.keywords,self.embeddings)
                self.saveSeedWordEmbeddings(seedEmbeddings)
                self.keywords.sort(reverse=True, key=lambda x : x[1])    
                self.saveKeywordsSortedToFile(self.keywords)
                self.size = len(self.keywords)
                
                return self.keywords

        def printKeyWords(self, QueryWord):                #util function to print keywords
                keywords = self.generateKeyWords(QueryWord)
                print("\nKeywords associated with the given query are - \n", keywords)
                # print("\n\n\n", self.embeddings[0])
                # print("\n\n\nshape of embeddings - ", np.shape(self.embeddings))
                # print("\n\n\nlen of each embedding - ", len(self.embeddings[0]))
                print("\n\nlen of keywords - ", len(keywords))
                print("\n\nlen of first keywords embedding vector - ", np.shape(self.embeddings))


# MyModel = SummaryModule()
# MyModel.printKeyWords("Covid Delhi")