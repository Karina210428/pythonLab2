import nltk as nl
nl.download('punkt')
import scipy as sc
from math import log
from sklearn.preprocessing import MinMaxScaler
from scipy.linalg import svd
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.decomposition import PCA
class LineDiscriminantAnalysis(object):
    def fit(self, x, y): 
        one_centroid = x[y==0].mean(axis = 0) 
        two_centroid = x[y==1].mean(axis = 0) 
        self.centroid = two_centroid - one_centroid 

    def predict(self, x): 
        return self.centroid.dot(x.T) 

    def printListTrain(self):
        print(self.listTrain)

    def y_exit(self): 
        scaler = MinMaxScaler() 
        X = self.centroid 
        X[X>=0.5]=1 
        X[X<0.5]=0 
        return X 

    def SUM(self, y, predict):
        sum = 0
        i=0
        X = []
        X = y.values == predict
        while i < len(y):
            if X[i] == True:
                sum+=1
            i+=1
        return sum

    def FP(self,y,predict):
        sum = 0
        j=0
        X = []
        while j< len(y):
            if y.values[j] == 0 and predict[j] == 1:
                sum+=1
            j+=1
        return sum

    def TP(self,y,predict):
        sum = 0
        j=0
        while j < len(y):
            if y.values[j] == 1 and predict[j] == 1:
                sum+=1
            j+=1
        return sum

    def FN(self,y,predict):
        sum = 0
        j=0
        X = []
        while j< len(y):
            if y.values[j] == 1 and predict[j] == 0:
                sum+=1
            j+=1
        return sum

    def plot_subfigure(self, bow,y):
        pca = PCA(n_components=2)
        X = pca.fit_transform(bow)

        spam = X[y==1]
        ham = X[y==0]
        plt.scatter(spam[:,0], spam[:,1], s=40, c='gray', edgecolors=(0, 0, 0))
        plt.scatter(ham[:,0], ham[:,1], s=40, c='r', edgecolors=(0, 0, 0))
        plt.show()

    def tokenize(self,texts): 
        return [nl.word_tokenize(text.lower()) for text in texts] 
        
    
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





