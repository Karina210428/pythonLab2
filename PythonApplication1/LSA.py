import nltk.stem as ps
import scipy as sc
from math import log
from scipy.linalg import svd
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.decomposition import PCA
class LSA(object):
    def __init__(self, stopwords, ignorechars):
        self.wdict=[]
        self.dictionary = []
        self.ignorechars=ignorechars
        self.stopwords=stopwords
        self.dcount=0

    def parse(self, doc):
        words = doc.split();
        self.listTrain=[]
        for w in words:
            t = w.maketrans("","",self.ignorechars)
            w = w.lower().translate(t)
            p = ps.PorterStemmer()
            w = p.stem(w)#приводим слово к начальной форме
            if w in self.stopwords:
                continue
            elif w in self.wdict:
               self.listTrain.append(w)
        return self.listTrain

    def printDict(self):
        print(self.wdict)

    def printListTrain(self):
        print(self.listTrain)

    def createDictionary(self, label_train):
        for index, item in enumerate(label_train, start=0):
            if(item=='spam'):
                for w in self.listTrain[index]:
                    self.wdict.append(w)
        self.counts = list(Counter(self.wdict))
        print(self.counts)

    def buildMatrix(self):
        self.keys = [k for k in self.wdict.keys() if len(self.wdict[k])>0]
        self.keys.sort();
        self.A = sc.zeros([len(self.keys),self.dcount])
        for i, k in enumerate(self.keys):
            for d in self.wdict[k]:
                self.A[i,d]=1

    def plot_subfigure(self, X):
        X = PCA(n_components=2).fit_transform(X)

       # plt.scatter(-0.5,-0.5,X,c="R")
        plt.scatter(X[:, 0], X[:, 1], s=40, c='gray', edgecolors=(0, 0, 0))
        plt.show()
        
    
    def printA(self):
        print(self.A)

    def calc(self):
        self.U, self.S, self.Vt = svd(self.A)

    def TFIDF(self):
        wordsPerDoc = sum(self.A, axis = 0)
        docsPerWord = sum(asarray(self.A > 0, 'i'), axis = 1)
        rows,cols = self.A.shape
        for i in range(rows):
            for j in range(cols):
                self.A[i,j] = (self.A[i,j]/ wordsPerDoc[j]) *log(float(cols)/ docsPerWord[i])
  
    def printSVD(self):
        print ('Here is the S matrix')
        print(self.S)
        print ('Here is the U matrix')
        print(-1*self.U[:,0:3])
        print ('Here is the Vt matrix')
        print(-1*self.Vt[0:3,:])





