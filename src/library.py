#############################################################################################
#########   les libraries nécessaires                                 #######################
#############################################################################################

import datetime
import numpy as np 
import random
import pandas as pd
from sklearn.cluster import KMeans
from scipy.stats import truncnorm
from sklearn.model_selection import GridSearchCV,train_test_split,ShuffleSplit,LeaveOneOut,cross_val_score
from sklearn.metrics import silhouette_score
from sklearn.tree import DecisionTreeClassifier , DecisionTreeRegressor
from scipy.cluster import hierarchy
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
import seaborn as sns
import networkx as nx
import plotly
from networkx.drawing.nx_agraph import graphviz_layout
from itertools import combinations
import warnings
warnings.filterwarnings('ignore')
import pydot
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering

