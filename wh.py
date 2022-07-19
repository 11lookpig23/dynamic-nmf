## input dynamic W,H
import joblib
import numpy as np
from collections import Counter
import heapq
import pickle
import pandas as pd
num_topics = 670
years = 12
path_dyn = 'out/dynamictopics_k'+str(num_topics)+'.pkl'

### 1. Read
def load_nmf_results( in_path ):
	(doc_ids, terms, term_rankings, partition, W, H, labels) = joblib.load( in_path )
	return (doc_ids, terms, term_rankings, partition, W, H, labels)

(doc_ids, terms, term_rankings, partition, W, H, labels) = load_nmf_results(path_dyn)

with open('data/embeddings.npy','rb') as f:
    embed = np.load(f,allow_pickle=True)
with open('data/vocabulary.pkl','rb') as f:
    vocab = pickle.load(f)

# 2. 词典转换
vocalist = list(vocab.keys())
df = pd.DataFrame(data = H, 
                  columns = terms)
Hre = df[vocalist].to_numpy()

# 3. 求逆矩阵，解新W
B = np.linalg.pinv(Hre)
Wli = []
for i in range(years):
    W_ = np.matmul(embed[i].toarray(),B)
    Wli.append(W_)

# 4. get partition
def generate_partition(W):
    return np.argmax( W, axis = 1 ).flatten().tolist()
    ## 可进一步改进

def generate_partition2(W):
    tot = [0]*num_topics
    W[W < 0.0368] = 0
    for w in W:
        nlar = heapq.nlargest(5, enumerate(w), key=lambda x: x[1])
        if nlar[0][1] == 0:
            continue
        else:
            delt = [(nlar[0][1] - nlar[i+1][1]) for i in range(4)]
            adli = [nlar[0]]#[nlar[0][0]]
            for i in range(4):
                d = delt[i]
                if d<0.0274:
                    adli.append(nlar[i+1])
            s = sum([a[1] for a in adli])
            for a in adli:
                tot[a[0]] += a[1]/s#1/len(adli)
    return [round(t) for t in tot]

def generate_partition3(W):
    tot = [0]*num_topics
    W[W < 0.0368] = 0
    for w in W:
        nlar = heapq.nlargest(5, enumerate(w), key=lambda x: x[1])
        if nlar[0][1] == 0:
            continue
        else:
            delt = [(nlar[0][1] - nlar[i+1][1]) for i in range(4)]
            adli = [nlar[0]]#[nlar[0][0]]
            for i in range(4):
                d = delt[i]
                if d<0.0274:
                    adli.append(nlar[i+1])
            s = sum([a[1] for a in adli])
            for a in adli:
                tot[a[0]] += a[1]/s#1/len(adli)
    return [round(t) for t in tot]

def time_topic(dicw):
    dic = dict.fromkeys(range(num_topics),0)
    for k in dicw.keys():
        dic[k] = dicw[k]
    return dic

tot = {}
for i in range(years):
    par = generate_partition2(Wli[i])
    dicw = Counter(par)
    dicw = time_topic(dicw)
    tot[i+2006] = dicw
totdf = pd.DataFrame(data = tot)
totdf.save("res/tot.csv")
np.save("res/Wn.npy",Wli)
    #res2 = dict(sorted(dicw.items(), key = itemgetter(1), reverse = True)[:])
    #res2
    ## return : pandas