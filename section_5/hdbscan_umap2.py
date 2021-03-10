import numpy as np
import pandas as pd 
from hdbscan import HDBSCAN
import umap
import seaborn as sns
import matplotlib.pyplot as plt
#import plotly.graph_objs as go
from sklearn.metrics import completeness_score, homogeneity_score, v_measure_score
from matplotlib.lines import Line2D

# descion tree
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_graphviz # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
from six import StringIO  
from IPython.display import Image  
import pydotplus

# tsne
from sklearn.manifold import TSNE

# random forest
from sklearn.ensemble import ExtraTreesClassifier

# colours
from matplotlib.colors import rgb2hex

from collections import Counter

# description of distance metrics here:
# https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.DistanceMetric.html

plot_kwds = {'alpha' : 0.5, 's' : 80, 'linewidths':0}

mcs = 10 # smallest cluster size
metric = 'euclidean'

colour_set_rgb = [[2,63,165],[125,135,185],[190,193,212],[214,188,192],[187,119,132],[142,6,59],[74,111,227],[133,149,225],[181,187,227],[230,175,185],[224,123,145],[211,63,106],[17,198,56],[141,213,147],[198,222,199],[234,211,198],[240,185,141],[239,151,8],[15,207,192],[156,222,214],[213,234,231],[243,225,235],[246,196,225],[247,156,212]]
        
colour_set = [[j / 255 for j in x] for x in colour_set_rgb]
colour_set = [ rgb2hex(i) for i in colour_set]


## read in the data
## these are within language distance matrices - not between language 
string_df = pd.read_csv("./section_5/processed_data/g0_relativeage_uniqueglottocodes_idsrem_dist.csv") #

string_mat = string_df[string_df.columns[:-1]]
print(string_mat.shape)

# possible distance metrics
# ['euclidean', 'l2', 'l1', 'manhattan', 'cityblock', 'braycurtis', 'canberra', 'chebyshev', 
# 'correlation', 'cosine', 'dice', 'hamming', 'jaccard', 'kulsinski', 'mahalanobis', metric, 
# 'minkowski', 'rogerstanimoto', 'russellrao', 'seuclidean', 'sokalmichener', 'sokalsneath', 'sqeuclidean', 
# 'yule', 'wminkowski']

# what settings give the least outliers
settings = []
for x in range(3, 12):  # check from 3 to 11. 3 is HDBSCAN minimum (below is single chain linking)
                        # 11 was shown to decrease performance in simulations.
    clusterer = HDBSCAN(algorithm='best', alpha=1.0,
        approx_min_span_tree=True,
        gen_min_span_tree=False,
        leaf_size=100,
        metric=metric,
        min_cluster_size=mcs, # smallest size a cluster can be
        min_samples=x, # how conservative should clustering be (larger = more conservative)
                        # 7 gives the lowest number of outliers.
        p=None,
        cluster_selection_method='eom')
    clusterer.fit(string_mat)
    cluster_count = list(np.unique(clusterer.labels_, return_counts=True))
    cluster_count = pd.DataFrame(np.transpose(cluster_count), columns = ['Group', 'Count'])
    print(x, cluster_count.iloc[0,1])
    settings.append([x, cluster_count.iloc[0,1]])

settings = pd.DataFrame(settings)
settings.columns = ['value', 'outliers']
settings = settings.loc[settings['outliers'].idxmin()]
ms = int(settings['value'])
# 
# 
# # In[3]:
# 
# 
# print(ms)
# 
clusterer = HDBSCAN(algorithm='best', alpha=1.0,
    approx_min_span_tree=True,
    gen_min_span_tree=True,
    leaf_size=100,
    metric="matching",
    min_cluster_size=10, # smallest size a cluster can be
    min_samples=3, # how conservative should clustering be (larger = more conservative)
    p=None,
    cluster_selection_method='eom')
clusterer.fit(string_mat)
# 
#clusterer.single_linkage_tree_.plot(cmap='viridis', colorbar=True)
cluster_count = list(np.unique(clusterer.labels_, return_counts=True))
cluster_count = pd.DataFrame(np.transpose(cluster_count), columns = ['Group', 'Count'])
print(cluster_count)

# # organise cluster output
output = pd.DataFrame(columns=['Language_ID','label_'])
output['Language_ID'] = string_df.Language_ID
output['label_'] = clusterer.labels_.tolist()
output['outlier_prob'] = clusterer.outlier_scores_.tolist()
output['cluster_prob'] = clusterer.probabilities_.tolist()
#
# ignore outliers
indices = output.label_ != -1
output = output[indices]
strmat_subset = string_mat[indices]

x_train, x_test, y_train, y_test = train_test_split(strmat_subset, output['label_'], test_size=0.25, random_state=1) # 75% training and 25% test

# Create Decision Tree classifer object
unique_groups = np.unique(clusterer.labels_)
clf = DecisionTreeClassifier(criterion="entropy", max_leaf_nodes = len(unique_groups)-1)

# Train Decision Tree Classifer
clf = clf.fit(x_train,y_train)

#Predict the response for test dataset
y_pred = clf.predict(x_test)

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

## random forest
forest = ExtraTreesClassifier(n_estimators=1000,
                          random_state=0)

forest.fit(strmat_subset, output['label_'])
importances = forest.feature_importances_
std = np.std([tree.feature_importances_ for tree in forest.estimators_],
           axis=0)
indices = np.argsort(importances)[::-1]

feature_importance = pd.DataFrame(columns = ['feature', 'importance'])
feature_importance['feature'] = indices
feature_importance['importance'] = importances[indices]

# Plot the feature importances of the forest
# plt.figure()
# plt.title("Feature importances")
# plt.bar(range(string_mat.shape[1]), importances[indices],
#        color="r", yerr=std[indices], align="center")
# plt.xticks(range(string_mat.shape[1]), indices)
# plt.xlim([-1, string_mat.shape[1]])
#plt.show()

print("Feature ranking:")
for f in range(0, len(string_mat.columns)):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

## Important features
most_important = feature_importance.importance >= 0.002156

FI_mat = string_mat.iloc[:,feature_importance.feature[most_important]]
settings=[]
for x in range(3, 12):
    clusterer = HDBSCAN(algorithm='best', alpha=1.0,
        approx_min_span_tree=True,
        gen_min_span_tree=True,
        leaf_size=100,
        metric=metric,
        min_cluster_size=10, # smallest size a cluster can be
        min_samples=x, # how conservative should clustering be (larger = more conservative)
                        # 7 gives the lowest number of outliers.
        p=None,
        cluster_selection_method='eom')
    clusterer.fit(FI_mat)
    cluster_count = list(np.unique(clusterer.labels_, return_counts=True))
    cluster_count = pd.DataFrame(np.transpose(cluster_count), columns = ['Group', 'Count'])
    print(x, cluster_count.iloc[0,1])
    settings.append([x, cluster_count.iloc[0,1]])

settings = pd.DataFrame(settings)
settings.columns = ['value', 'outliers']
settings = settings.loc[settings['outliers'].idxmin()]
ms = int(settings['value'])

clusterer = HDBSCAN(algorithm='best', alpha=1.0,
    approx_min_span_tree=True,
    gen_min_span_tree=True,
    leaf_size=100,
    metric="matching",
    min_cluster_size=10, # smallest size a cluster can be
    min_samples=3, # how conservative should clustering be (larger = more conservative)
    p=None,
    cluster_selection_method='eom')
clusterer.fit(FI_mat)


cluster_count = list(np.unique(clusterer.labels_, return_counts=True))
cluster_count = pd.DataFrame(np.transpose(cluster_count), columns = ['Group', 'Count'])
#cluster_count

# reorganise cluster output
output = pd.DataFrame(columns=['Language_ID','label_'])
output['Language_ID'] = string_df.Language_ID
output['label_'] = clusterer.labels_.tolist()
output['outlier_prob'] = clusterer.outlier_scores_.tolist()
output['cluster_prob'] = clusterer.probabilities_.tolist()

output.to_csv('./section_5/results/g0.csv')



# remove uncategorised langauges from the decision tree
indices = output.label_ != -1
output2 = output[indices]
FI_mat2 = FI_mat[indices]

x_train, x_test, y_train, y_test = train_test_split(FI_mat2, output2['label_'], test_size=0.25, random_state=1) # 75% training and 25% test

# Create Decision Tree classifer object
unique_groups = np.unique(clusterer.labels_)
clf = DecisionTreeClassifier(criterion="entropy", max_leaf_nodes = len(unique_groups)-1)

# Train Decision Tree Classifer
clf = clf.fit(x_train,y_train)

#Predict the response for test dataset
y_pred = clf.predict(x_test)

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

## plot tree
# # import graphviz
# # from sklearn.tree import export_graphviz
#
# # graphviz.Source(export_graphviz(clf))
# max_class = np.max(output2['label_'])
# class_names = [str(x) for x in (range(0, max_class+1))]
# print(class_names)
# dot_data = StringIO()
# export_graphviz(clf, out_file=dot_data,
#                 filled=True, rounded=True,
#                 special_characters=True, feature_names = FI_mat2.columns, class_names = class_names)
# graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
#
# graph.write_pdf("./section_5/results/decision-trees/g0.pdf")

#Image(graph.create_png())

label = [clusterer.labels_]
color_palette = sns.color_palette('Paired', 47)
cluster_colors = [color_palette[x] if x >= 0
                  else (0.5, 0.5, 0.5)
                  for x in clusterer.labels_]
cluster_member_colors = [sns.desaturate(x, p) for x, p in
                         zip(cluster_colors, clusterer.probabilities_)]

## legend creation
unique_groups = np.unique(clusterer.labels_)
unique_colours = [color_palette[x] if x >= 0
                  else (0.5, 0.5, 0.5)
                  for x in unique_groups]

custom_scatter = []
for i in range(0, len(unique_groups)):
    custom_scatter.append(Line2D([0], [0], marker='o', color=unique_colours[i], label=str(unique_groups[i]),
                          markerfacecolor=unique_colours[i], markersize=10, lw=0))

reducer = umap.UMAP(
n_neighbors=200, # low numbers emphasise local structure - high numbers global structure
    min_dist=0.1, # how tight are values clustered (low = tighter; high = looser)
    n_components=2, # number of dimensions to output
    random_state=42)
embedding = reducer.fit_transform(string_mat)


# n_neighbors=20, # low numbers emphasise local structure - high numbers global structure
#     min_dist=2, # how tight are values clustered (low = tighter; high = looser)
#     n_components=2, # number of dimensions to output
#     random_state=42,  # set random seed
#     spread = 2
# fig, ax = plt.subplots()
# jittered_y = embedding[:,1] + 0.2 * np.random.rand(len(embedding[:,1])) -0.05
# jittered_x = embedding[:,0] + 0.2 * np.random.rand(len(embedding[:,0])) -0.05
# ax.scatter(x = jittered_x, y = jittered_y, s=50, linewidth=0, c=cluster_colors, alpha=0.8)
# ax.legend(handles=custom_scatter, loc='lower right', ncol = 3)
# plt.show()

#fig.savefig(DIR+'results/umap/g0.pdf', format='pdf')

# save embeddings
np.savetxt('section_5/results/g0_embeddings.csv', embedding, delimiter=",")

print(Counter(output['label_']))
