import numpy as np
class BOW(object): 

    def fit(self,texts): 
        tmp =[] 
        for words in texts: 
            tmp.extend(words) 
            self.dictionary = set(tmp) 
            self.word_index = {} 
            self.index_word = {} 

        for i,v in enumerate(self.dictionary): 
            self.word_index[v]= i 
            self.index_word[i]= v 


    def transform (self,texts): 
        self.bow = [] 
        for words in texts: 
            vec = [0] * (len(self.dictionary)) 
            for word in words: 
                if word in self.word_index: 
                    vec[self.word_index[word]]=1 
            self.bow.append(vec) 
        return np.array(self.bow)

    def TFIDF(self):
        wordsPerDoc = sum(self.bow, axis = 0)
        docsPerWord = sum(asarray(self.bow > 0, 'i'), axis = 1)
        rows,cols = self.bow.shape
        for i in range(rows):
            for j in range(cols):
                self.bow[i,j] = (self.bow[i,j]/ wordsPerDoc[j]) *log(float(cols)/ docsPerWord[i])
        return self.bow