# ui_app.py
from flask import Flask, render_template, request, jsonify
import random
import string
from matplotlib import pyplot as plt 
import numpy as np
import base64
from sklearn.cluster import KMeans
from io import BytesIO
import random 
import string 
import pandas as pd
import networkx as nx
from collections import Counter
app = Flask(__name__)

tweets={}
retweets=[]
mentions={}
@app.route('/')
def index():
    return render_template('index.html')
@app.route('/network',methods=['POST'])
def network():
    df=pd.read_csv('data.csv')
    random_id =df['id'].iloc[-1]
    sender=request.form['sender']
    receiver=request.form['receiver']
    messege=request.form['message']
    id=random_id+1
    data={'id':id,'sender':sender,'receiver':receiver,'messege':messege}
    df=df.append(data,ignore_index=True)
    df = df.drop_duplicates(subset='id')
    df.to_csv('data.csv',index=False)
    # new_row1=np.array([['jjjjjjjj','eeeeeeeee'],['jjjjjjjj','eeeeeeeee']])
    # new_row2=np.array([['sssssssssss','jjjjjjjj'],['sssssssssss','jjjjjjjj']])
    new_in_set = np.array(df.iloc[:,[1,2]])
    graph = nx.Graph()
    all_users = list(set(new_in_set[:,0]) | set(new_in_set[:,1]))
    graph.add_nodes_from(all_users, count=10)
    node_colours = []
    for node in graph:
        if node in (set(new_in_set[:,0]) & set(new_in_set[:,1])):
            node_colours.append('g')
        elif node in np.unique(new_in_set[:,0]):
            node_colours.append('r')
        elif node in np.unique(new_in_set[:,1]):
            node_colours.append('b')
    edges = {}
    occurrence_count = Counter(map(tuple, new_in_set))
    for (sender, receiver), count in occurrence_count.items():
        if (receiver, sender) in edges.keys():
            edges[(receiver, sender)] = edges[(receiver, sender)] + count
        else:
            edges[(sender, receiver)] = count
    for (sender, receiver), count in edges.items():
        graph.add_edge(sender, receiver, weight=count)
    plt.figure(figsize=(10,10))
    nx.draw(graph, pos=nx.spring_layout(graph),node_color=node_colours, with_labels=True)
    img1 = BytesIO()
    plt.savefig(img1, format='png')
    img1.seek(0)
    img_base641 = base64.b64encode(img1.read()).decode()
    img1.close()
# plt.show()
    np.random.seed(42)
    X = np.random.rand(100, 2)
    kmeans = KMeans(n_clusters=3, random_state=42)
    labels = kmeans.fit_predict(X)
    plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis')
    plt.title('Clustering Output')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')

    img = BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    img_base642 = base64.b64encode(img.read()).decode()
    img.close()
    return render_template('index.html',img_base641=img_base641)#img_base642=img_base642)
if __name__ == '__main__':
    app.run(debug=True,port=5000,host='0.0.0.0')
