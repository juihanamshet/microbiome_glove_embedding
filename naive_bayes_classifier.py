import pandas as pd 
import numpy as np 
import pickle
import os
import helper_functions as hf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB 

data_dir = "C:/Users/juiha/OneDrive - Drexel University/Drexel/terms/term11/ECES 487/project/clustering_embeddings/microbiome_glove_embedding/data"
fig_dir = "C:/Users/juiha/OneDrive - Drexel University/Drexel/terms/term11/ECES 487/project/clustering_embeddings/microbiome_glove_embedding/figures"

f = open(os.path.join(data_dir, "X_sample_property.obj"), "rb")
X_embed = pickle.load(f)
f.close()

f = open(os.path.join(data_dir, "y_sample_ibd.obj"), "rb")
y_embed = pickle.load(f)
f.close()

# loading training data
# sample by taxa 
f = open(os.path.join(data_dir, "otu_train_.07.obj"), "rb")
otu_train = pickle.load(f)
f.close()

# loading test data
# sample by taxa
f = open(os.path.join(data_dir, "otu_test_.07.obj"), "rb")
otu_test = pickle.load(f)
f.close()

# sample by 13 feature vectors (IBD, exercise etc)
f = open(os.path.join(data_dir, "map_test_.07.obj"),"rb")
map_test = pickle.load(f)
f.close()

# sample by 13 feature vectors (IBD, exercise etc)
f = open(os.path.join(data_dir, "map_train_.07.obj"),"rb")
map_train = pickle.load(f)
f.close()

y_embed = list(y_embed[0].values)
# Imbalanced data set: only 317 samples have IBD out of 5874

# Classifying embedded data i.e. 113 features using Naive Bayes
X_embed_train, X_embed_test, y_embed_train, y_embed_test = train_test_split(X_embed, y_embed, test_size = 0.2, random_state = 10)

# Input data has negative values, MultinomialNB and ComplementNG cannot be used. 
clf = GaussianNB()
model = clf.fit(X_embed_train, y_embed_train)

predicted_y = model.predict(X_embed_test)

f = plt.figure(figsize=(15,5))
roc_auc, fpr, tpr, average_precision, f1, f2 = hf.computeMLstats(model, X_embed_test, y_embed_test, plot=True, plot_pr=True, graph_title = "Naive Bayes Classifier on embedded data", flipped = False)
f.savefig(os.path.join(fig_dir, "naive_bayes_classifier_embed.pdf"))

# Classifying OTU data i.e. 26k+ features using Naive Bayes
X_train, X_val, X_test, y_train, y_val, y_test = hf.getMlInput(otu_train, otu_test, map_train, map_test, target = "IBD", asinNormalized=True)
X_train = pd.concat([X_train, X_val], axis = 0)
y_train = y_train + y_val
 
# Input data has negative values, MultinomialNB and ComplementNG cannot be used. 
clf = GaussianNB()
model = clf.fit(X_train, y_train)

predicted_y = model.predict(X_test)

f = plt.figure(figsize=(15,5))
roc_auc, fpr, tpr, average_precision, f1, f2 = hf.computeMLstats(model, X_test, y_test, plot=True, plot_pr=True, graph_title = "Naive Bayes Classifier on OTU table", flipped = False)
f.savefig(os.path.join(fig_dir, "naive_bayes_classifier_otu.pdf"))




