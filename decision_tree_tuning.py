import pandas as pd 
import numpy as np 
import pickle
import os
import helper_functions as hf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB 
from sklearn import tree

data_dir = "C:/Users/juiha/OneDrive - Drexel University/Drexel/terms/term11/ECES 487/project/clustering_embeddings/microbiome_glove_embedding/data"
fig_dir = "C:/Users/juiha/OneDrive - Drexel University/Drexel/terms/term11/ECES 487/project/clustering_embeddings/microbiome_glove_embedding/figures"

# f = open(os.path.join(data_dir, "X_sample_property.obj"), "rb")
# X_embed = pickle.load(f)
# f.close()

# f = open(os.path.join(data_dir, "y_sample_ibd.obj"), "rb")
# y_embed = pickle.load(f)
# f.close()

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

# Classifying OTU data i.e. 26k+ features using Decision Trees
X_train, X_val, X_test, y_train, y_val, y_test = hf.getMlInput(otu_train, otu_test, map_train, map_test, target = "IBD", asinNormalized=True)
X_train = pd.concat([X_train, X_val], axis = 0)
y_train = y_train + y_val

max_depths = np.linspace(1,32,32,endpoint=True)
train_results = []
test_results = []
for max_depth in max_depths:
    print(max_depth)
    model = tree.DecisionTreeClassifier(max_depth=max_depth)
    model.fit(X_train, y_train)

    roc_auc, fpr, tpr, average_precision, f1, f2 = hf.computeMLstats(model, X_test, y_test, plot=True, plot_pr=True, graph_title = "Decision Tree Classifier on Embedded Data", flipped = False)

    test_results.append(roc_auc)
    print("appended", roc_auc)

    roc_auc, fpr, tpr, average_precision, f1, f2 = hf.computeMLstats(model, X_train, y_train, plot=True, plot_pr=True, graph_title = "Decision Tree Classifier on Embedded Data", flipped = False)

    train_results.append(roc_auc)
    print("appended", roc_auc)

from matplotlib.legend_handler import HandlerLine2D
line1, = plt.plot(max_depths, train_results, 'b', label='Train AUC')
line2, = plt.plot(max_depths, test_results, 'r', label='Test AUC')
plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
plt.ylabel('AUC score')
plt.xlabel('Tree depth')
plt.show()

max_index = test_results.index(max(test_results))
best_max_depth = max_index + 1
print(best_max_depth)

min_samples_splits = np.linspace(0.1, 1, 10, endpoint=True)
train_results = []
test_results = []

for min_samples_split in min_samples_splits:
    model = tree.DecisionTreeClassifier(min_samples_split=min_samples_split)
    model.fit(X_train, y_train)

    roc_auc, fpr, tpr, average_precision, f1, f2 = hf.computeMLstats(model, X_test, y_test, plot=True, plot_pr=True, graph_title = "Decision Tree Classifier on Embedded Data", flipped = False)

    test_results.append(roc_auc)

    roc_auc, fpr, tpr, average_precision, f1, f2 = hf.computeMLstats(model, X_train, y_train, plot=True, plot_pr=True, graph_title = "Decision Tree Classifier on Embedded Data", flipped = False)

    train_results.append(roc_auc)

from matplotlib.legend_handler import HandlerLine2D
line1, = plt.plot(min_samples_splits, train_results, 'b', label='Train AUC')
line2, = plt.plot(min_samples_splits, test_results, 'r', label='Test AUC')
plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
plt.ylabel('AUC score')
plt.xlabel('min samples split')
plt.show()

max_index = test_results.index(max(test_results))
best_min_sample_split = min_samples_splits[max_index]
print(best_min_sample_split)

min_samples_leafs = np.linspace(0.1, 0.5, 5, endpoint=True)
train_results = []
test_results = []
for min_samples_leaf in min_samples_leafs:
    model = tree.DecisionTreeClassifier(min_samples_leaf=min_samples_leaf)
    model.fit(X_train, y_train)

    roc_auc, fpr, tpr, average_precision, f1, f2 = hf.computeMLstats(model, X_test, y_test, plot=True, plot_pr=True, graph_title = "Decision Tree Classifier on Embedded Data", flipped = False)

    test_results.append(roc_auc)

    roc_auc, fpr, tpr, average_precision, f1, f2 = hf.computeMLstats(model, X_train, y_train, plot=True, plot_pr=True, graph_title = "Decision Tree Classifier on Embedded Data", flipped = False)

    train_results.append(roc_auc)

from matplotlib.legend_handler import HandlerLine2D
line1, = plt.plot(min_samples_leafs, train_results, 'b', label='Train AUC')
line2, = plt.plot(min_samples_leafs, test_results, 'r', label='Test AUC')
plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
plt.ylabel('AUC score')
plt.xlabel('min samples leaf')
plt.show()

max_index = test_results.index(max(test_results))
best_min_sample_leaf = min_samples_leafs[max_index]
print(best_min_sample_leaf)

model = tree.DecisionTreeClassifier(max_depth=best_max_depth, min_samples_split=best_min_sample_split, min_samples_leaf=best_min_sample_leaf)
model.fit(X_train, y_train)

f = plt.figure(figsize=(15,5))
roc_auc, fpr, tpr, average_precision, f1, f2 = hf.computeMLstats(model, X_test, y_test, plot=True, plot_pr=True, graph_title = "Decision Tree Classifier on OTU Data", flipped = False)
f.savefig(os.path.join(fig_dir, "decision_tree_classifier_otu_tuned.pdf"))