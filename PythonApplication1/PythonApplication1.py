import pandas as pd
from LDA import LineDiscriminantAnalysis
from BOW import BOW
import nltk 
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
import numpy as np
from sklearn.model_selection import train_test_split

dataset = pd.read_csv('spam.csv', encoding ='ansi')
print(len(dataset))
dataset = dataset[int(len(dataset)*0.2):]
print(len(dataset))
dataset = dataset[['v1', 'v2']] 
dataset['v1'] = dataset['v1'] == 'spam' 
dataset['v1'] = dataset['v1'].astype(np.int32) 
print(dataset.head(20))
mylda = LineDiscriminantAnalysis()
texts=mylda.tokenize(dataset['v2']) 
y = dataset['v1']

vectorizer = BOW()
vectorizer.fit(texts)
#print(vectorizer.dictionary)
#print(vectorizer.word_index) 
#print(vectorizer.index_word) 
#vectorizer.transform(texts) 
bow = vectorizer.transform(texts) 
#for i, v in enumerate(bow[0]): 
 #   if v ==i: 
  #      print(vectorizer.index_word[i])
#print(bow)
mylda.plot_subfigure(bow,y)

#vectorizer1 = BOW()
#vectorizer1.fit(texts)
#vectorizer1.transform(texts) 
#bow1 = vectorizer1.transform(texts) 
#bow2 = vectorizer1.TFIDF()

#bow.mean(axis = 0)
 
mylda.fit(bow,y) 
mylda.predict(bow) 
predict = [int(v) for v in mylda.predict(bow) > 0] 
#print(y.tolist())


sum  = mylda.SUM(y, predict)
print('Accuracy1')
print(sum/len(y))
print('Accuracy2')
print(accuracy_score(y,predict)) 

precision = mylda.TP(y,predict)/(mylda.TP(y,predict)+mylda.FP(y,predict))
print('Precision1')
print(precision)
print('Precision2')
print(precision_score(y,predict))

recall = mylda.TP(y,predict)/(mylda.TP(y,predict)+mylda.FN(y,predict))
print('Recall_1')
print(recall)
print('Recall_2')
print(recall_score(y,predict)) 

F1 = 2*(precision*recall)/(precision+recall)
print('F1_1')
print(F1)
print('F1_2')
print(f1_score(y,predict)) 



#print(df_train.head(5))

#mylsa = LSA(stopwords,ignorechars)
#for t in df_train:
 #   mylsa.parse(t)
#mylsa.printListTrain()
#mylsa.createDictionary(label_train)
#mylsa.printDict()

#mylsa.buildMatrix()
#mylsa.printA()
#X = mylsa.printA()
#mylsa.plot_subfigure(X[X['v1']=='spam'])
#mylsa.calc()
#mylsa.printSVD()

#text_processor = generate_processor(keep_alpha_only=True,stemmer_langs=["english"])
#docs_factory = lambda: df.sms.words(keep_levels=Levels.Nothing, **text_processor)
#word = Counter((word for doc in docs_factory() for word in doc))
#print(word)


