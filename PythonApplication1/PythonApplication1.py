import pandas as pd
from LSA import LSA
import nltk 
from sklearn.model_selection import train_test_split
feature_dict = {i:label for i, label in zip(
    range(4),
    ('answ','sms','1','2',) ) }
df = pd.read_csv("spam.csv",sep=',', encoding='latin1',)
df.drop(df.columns[[2,3,4]],axis=1, inplace=True)
#df.columns = [l for i, l in sorted(feature_dict.items())]+ ['3']
#df.sms = df.sms.str.lower().traslate(None, self.ignorechars);
print(df.head(5))

df_train, df_test, label_train, label_test = train_test_split(df['v2'],df['v1'],test_size = 0.2)
stopwords = ['and','edition','for','in','little','of','the','to']
ignorechars = ''',:'.-&)?[]{}\/"0123456789!='''
mylsa = LSA(stopwords,ignorechars)
for t in df_train:
    mylsa.parse(t)
mylsa.printListTrain()
mylsa.createDictionary(label_train)
mylsa.printDict()

mylsa.buildMatrix()
mylsa.printA()
#X = mylsa.printA()
#mylsa.plot_subfigure(X[X['v1']=='spam'])
#mylsa.calc()
#mylsa.printSVD()

#text_processor = generate_processor(keep_alpha_only=True,stemmer_langs=["english"])
#docs_factory = lambda: df.sms.words(keep_levels=Levels.Nothing, **text_processor)
#word = Counter((word for doc in docs_factory() for word in doc))
#print(word)


