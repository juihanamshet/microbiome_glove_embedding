import pandas as pd 
import numpy as np 
import pickle
import os
import helper_functions as hf
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns

data_dir = "C:/Users/juiha/OneDrive - Drexel University/Drexel/terms/term11/ECES 487/project/clustering_embeddings/microbiome_glove_embedding/data"
fig_dir = "C:/Users/juiha/OneDrive - Drexel University/Drexel/terms/term11/ECES 487/project/clustering_embeddings/microbiome_glove_embedding/figures"

f = open(os.path.join(data_dir, "X_sample_property.obj"), "rb")
X = pickle.load(f)
f.close()

f = open(os.path.join(data_dir, "y_sample_ibd.obj"), "rb")
y = pickle.load(f)
f.close()

tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
tsne_results = tsne.fit_transform(X)

X['dim1'] = tsne_results[:,0]
X['dim2'] = tsne_results[:,1]
X['y'] = y

f = plt.figure(figsize=(16,10))
sns.scatterplot(
    x="dim1", y="dim2",
    hue="y",
    palette=sns.color_palette("hls", 2),
    data=X,
    legend="full",
    alpha=0.3
)

f.savefig(os.path.join(fig_dir, "tsne_embed.pdf"))



