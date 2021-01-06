import math

class tfidf:
    def __init__(self, df, n, tf):
        self.tf = tf
        self.df = df
        self.n = n
    
    def runtfidf(self):
        idf = {}
        for k,v in self.df.items():
            idf[k] = math.log10(self.n/v)
        for k,v in self.tf.items():
            for i,j in v.items():
                self.tf[k][i] = j*idf[i]
        return self.tf