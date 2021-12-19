from flask import Flask, render_template, request,g
import sqlite3 as sql
import os
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_matrix, csc_matrix, coo_matrix, lil_matrix

app = Flask(__name__)

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html', \
    title="相性のいい材料を教えます", \
    message="使いたい材料は？")

@app.route('/', methods=['POST','GET'])
def form():
    f=open('data/test.json','r')
    json_dict=json.load(f)
    vectorizer = TfidfVectorizer()

    docs = []
    for i in range(len(json_dict)):
        tmp_docs = [j.translate(str.maketrans({' ':'_','-':'_'})) for j in json_dict[i]['ingredients']]
        docs.append(' '.join(tmp_docs))
    vectorizer.fit(docs)
    X=vectorizer.transform(docs)
    wordlist=vectorizer.get_feature_names()

    field = request.form['field']
    lil = lil_matrix(X.toarray())
    coo=coo_matrix(X.toarray())

    idx2=[]
    for idx,food in enumerate(wordlist):
        if field==food:
            for j in range(9944):
                if X[j,idx] :
                    idx2.append(j)
                else: 
                    continue

    vl=[]
    for k in idx2:
        vl.append(lil.data[k])

    vl=[x for row in vl for x in row]
    new_vl=sorted(vl,reverse=True)

    a=[]
    for m in range(107810):
        for n in range(3):
            if coo.data[m]==new_vl[n]:
                a.append(wordlist[coo.col[m]])

    return render_template('index.html', \
    title="おすすめの材料教えます", \
    message="おすすめの材料は「%s」、「%s」、「%s」です!" % (a[0],a[1],a[2]))
        

if __name__ == '__main__':
    app.debug = True
    app.run(host='0,0,0,0', port=os.environ['PORT'])