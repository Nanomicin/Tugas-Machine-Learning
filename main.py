import csv
import preprocessing
import tfidf
from collections import Counter

def toDict(cont):
    d={}
    for k,v in cont.items():
        d[k] = v
    return d

with open('Indonesian Sentiment Twitter Dataset Labeled.csv', mode='r') as file :
    csv_reader = csv.DictReader(file)
    kelas = []
    tf = {}
    df = {}
    wtf = {}
    i=0    
    for row in csv_reader:
        text = f'{row["sentimen	Tweet"]}'
        text = text.lower()
        runpreproc = preprocessing.preprocessing()
        text = runpreproc.run_preproc(text)
        text = text.split()
        temp = text.pop(0)
        kelas.append(temp)
        
        count = Counter()
        count = count + Counter(text)
        f = toDict(count)
        for k in f.keys():
            if k not in df:
                df[k]=1
            else:
                df[k]+=1
        
        tf[i] = f
        i+=1

print(">> Finish Read File Indonesian Sentiment Twitter Dataset Labeled.csv and Preprocessing")

listterm = []
for v in tf.values():
    for k in v.keys():
        if k not in listterm:
            listterm.append(k)
listterm.sort()
for k,v in tf.items():
    for t in listterm:
        if t not in v.keys():
            tf[k][t] = 0

print(">> Finish Fill Empty Term")

n = i+1
bobot = tfidf.tfidf(df,n,tf)
wtf = bobot.runtfidf()

print(">> Finish Weighting TF-IDF")

with open('dataset.csv', 'a+', newline='') as file:
    kls = ['class']
    fnames = listterm + kls
    writer = csv.DictWriter(file, fieldnames=fnames)
    
    writer.writeheader()
    elementw = {}
    for k,v in wtf.items():
        elementw = v
        elementw['class'] = kelas[k]
        writer.writerow(elementw)

print(">> Finish Writing File dataset.csv")