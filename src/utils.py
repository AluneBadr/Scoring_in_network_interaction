###########################################################################
                    # libraries #
##########################################################################

import datetime
import numpy as np 
import random
import pandas as pd
from sklearn.cluster import KMeans
from scipy.stats import truncnorm
from sklearn.model_selection import GridSearchCV,train_test_split,ShuffleSplit,LeaveOneOut
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier 
from sklearn.utils import shuffle 
import matplotlib.pyplot as plt
import networkx as nx
import pydot
from networkx.drawing.nx_pydot import graphviz_layout
import warnings
warnings.filterwarnings('ignore')
from sklearn.datasets import make_blobs
perturbations = ['TRP', 'GEO', 'GRV','CR', 'AUTR','FIN', 'MP','LKS', 'ADM','INC']
step=8
############################################################################
            # some utils functions #
############################################################################

## aller à une date suivante en ajoutant un pas = 1 jour, 2 jours, 3jours, n jours...

def next_date(start_date, step) :
    start_date += datetime.timedelta(days=step)
    return start_date

## générer une loi normal tronquée
def get_truncated_normal(mean=0, sd=1, low=0, upp=10):
    return truncnorm(
        (low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)


############################################################################
            # create tree graph with n level and p predessessor #
############################################################################

def create_tree_graph(p=None,n=None,remove_first=False):
    
    '''
    Role : crée un graph aléatoire en partant sur un arbre puis en ajoutant des lien aléatoires.
    input : 
        -p : le nombre de parents (prédécesseurs) de chaque noeud
        -n : le nombre de niveau de l'arbre
        -remove_first : faut t-il garder le premier noeud de l'arbre ou pas
    output : un graph aléatoire
    '''
    Tree = nx.balanced_tree(p,n) # un graph de n niveaux. chaque noeud a p prédécesseurs
    adjency = pd.DataFrame(np.tril(nx.adjacency_matrix(Tree).todense().T))
    supliers_name = ['F'+str(i) for i in range(0,adjency.shape[1])]
    
    adjency.index = supliers_name
    adjency.columns = supliers_name
   
    Net = nx.from_pandas_adjacency(adjency , create_using=nx.DiGraph)
    
    # supprimer le premier noeud
    if remove_first==True:
        Net.remove_node('F0')
    
    #ajouter des liens aléatoires : on peu prendre k = n/3 pour plus d'aléa
    k = 0
    random.seed(12)
    while k < 4:
        Nodes = random.sample(set(list(Net.nodes)), 2)
        if (Net.has_edge(Nodes[0], Nodes[1])==False and Net.has_edge(Nodes[1], Nodes[0])==False):
            Net.add_edge(Nodes[0], Nodes[1])
            k+=1
        else:
            pass
    print("network is created succesfully...")
    return Net




############################################################################
            # get min, max, mean, median an std of predessessors #
############################################################################

def get_perf_predess(suplier_data=None,suplier=None,graph=None):
    
    '''
    Role : récupère les perfs moyens, medians... des parents d'un noeud
    input : 
        -suplier_data : la table contenant les données des fournisseurs.
        cette table aura une colonne "suplier" qui contient le nom des fournisseur et une colonne 'score' qui contient les score de
        performance
        - suplier : le nom du fournisseur en question
        - le graph d'interaction des fournisseurs 
    output :  retourne un dictionnaire avec les statistiques
    '''
    # tous les parents du noeud
    Predecess = [i for i in list(graph.predecessors(suplier))]
    
    if len(Predecess)==0 :
         res = {}
    else:
        
        vals = suplier_data.loc[suplier_data.supplier.isin(Predecess),"score"].tolist()
        
        res = {'mean_score_pred' : np.mean(vals), 
                "max_score_pred" : np.max(vals) , 
                "min_score_pred" : np.min(vals) , 
                "med_score_pred" : np.median(vals),
                "sd_score_pred"  : np.std(vals)}
        
    return res


 

def Normalize(X):
    return  X / 10




###############################################################################
                    #generate data#
###############################################################################

g = create_tree_graph(p=2,n=3,remove_first=False)
d = datetime.date(2020, 1, 1)

def generate_data(graph=g,nb_date=3,start_date=d,perturbations=['GEO','FIN','AUTR'],\
                  impacts = ['I1','I2'], Perf_parent=['mean','median'],centers=None, with_score = True, train=True):
    
    '''
    Role : A partir d'une instance de graph, générer toutes les caractéristiques de tous les noeuds (fournisseurs) sur plusieurs 
    dates. Ces caractéristiques sont les probas q'un fournisseurs subit un perturbations (GEO,FIN,AUTR) et les impacts mesurés sur 
    un score de 1 à 10.
    Inpurt : 
        -graph : un graph G
        -nb_date : le nombre de date (timestamp)
        -start_date : la date de la première observation
        - Perf_parent : les variable qui résument la performance des parents de chaque noeud
        - perturbations : list de toutes les perturbations potentielles
        - impacts : list de tous les impacts potentiels
        - with_score : si on veut générer les score de performance ou pas
        -train : si c'est une donnée train ou test
    Output : un dataframe
    '''
    data = pd.DataFrame()
    nb_supliers = len(graph.nodes)
    n_features = len(perturbations)+ len(impacts) + len(Perf_parent) + 1
    n = int(nb_supliers/10)  
    P = [n, n, n , 3*n/2, n, n/3, n/2,2*n, 2*n/3, nb_supliers-np.sum([n, n, n , 3*n/2, n, n/3, n/2,2*n, 2*n/3])]
    
    # ajustement de la dimention
    if np.sum([int(i) for i in P])==nb_supliers:
        pass
    else:
        P = [n+1, n, n , 3*n/2, n, n/3, n/2,2*n, 2*n/3, nb_supliers-np.sum([n, n, n , 3*n/2, n, n/3, n/2,2*n, 2*n/3])]
        
    print('generating diferent groupe of suppliers at each date....')

    for i in range(nb_date):
        np.random.seed(i)
        X, Y = make_blobs(n_samples=[int(i) for i in P], centers=centers, n_features=n_features, cluster_std=0.6,center_box=[0,10], random_state=i+2020)
        X = pd.DataFrame(X)
        Y = pd.DataFrame(Y)
        X.columns = perturbations + Perf_parent + impacts  + ['Cost_variation']
        
        for p in perturbations :
            X['{}'.format(p)] = Normalize(X['{}'.format(p)])
            
        X['Cost_variation']= Normalize(X['Cost_variation']) * np.random.choice([-1,1], size=(len(X)), p=[0.1, 1-0.1])
        
        
        X['min_score_pred'] = X[['max_score_pred','min_score_pred']].min(axis=1)
        X['max_score_pred'] = X[['max_score_pred','min_score_pred']].max(axis=1)

        X['supplier']  = list(graph.nodes)
        start_date = next_date(start_date,step=8)  
        X['date'] = start_date  
        data = pd.concat([data,X])
        
    data['score'] = 0
    
        # creation de la variable cible. Il s'agit de séparer les données en 5 groupe. Attribuer un score s compris entre 1 et 5
        #à chaque groupe en s'appuyant sur les centre de classe.
    print('generating scores...')
    
    for d in data.date.unique().tolist(): # pour chaque date
        Df = data[data.date==d] # je récup les données
        kmeans =  KMeans(n_clusters=len(P), max_iter=600, algorithm = 'auto', random_state=0) # je sépare en 5 groupe
        var= [i for i in Df.columns if i not in ['Unnamed: 0','supplier','date','score','cluster']]
        kmeans.fit(Df[var])
        center = pd.DataFrame(kmeans.cluster_centers_,columns=var)# je récup les centre de classe
        center['mean_risque'] = center[perturbations].mean(axis = 1) # je calcul la moyenne des perturbations. si elle est 
        #faible ça veux dire que ce groupe ne subit pas bcp de perturbations
        
        # je vais creer les scrores de performance en tenant compte de la perturbation moyenne et non toutes.
        # aussi sans tenir compte de impact pour ne pas trop influencer le model.
        
        #print(center.columns)
        #center['qlty_impact'] = center['qlty_impact'].rank()
        center['sd_score_pred'] = 1/center.sd_score_pred.rank()
        center['Cost_variation'] = 1/center.Cost_variation.rank()
        center['mean_risque'] = 1/center.mean_risque.rank()
        
        center = center[[i for i in center.columns if i not in perturbations+impacts]]
        
        for var in center.columns:
            center['{var}'.format(var = var+'_'+'rank')] = center['{name}'.format(name = var)].rank()
        
        # récupérer le ranking
        cols = [i for i in center if 'rank' in i]
        center['score'] = center[cols].mean(axis=1).rank(method='first')
        Df['label'] = kmeans.labels_
        Df['score'] = 0
        label = center.index.tolist()
        score = center.score.tolist()
        
        for i in label:
            Df.loc[Df.label==i,'score'] = score[i]
       
        # une fois les scores de perf créés, on peut mettre à jour les les variables Perf_parent
        for sup in Df.supplier:
            get_perf_predess(Df,sup,graph)
            
        data.loc[data.date==d,'score']=Df.score
        data.loc[data.date==d,"mean_score_pred"] = Df.mean_score_pred
        data.loc[data.date==d,"max_score_pred"] =  Df.max_score_pred
        data.loc[data.date==d,"min_score_pred"] =  Df.min_score_pred
        data.loc[data.date==d,"med_score_pred"] =  Df.med_score_pred
        data.loc[data.date==d,"med_score_pred"] =  Df.sd_score_pred
            
    
    
    np.random.seed(2020)
    coef = [.9,.11,.2,.4,.35,.73,.2,.8,.4,.1]
    
    print('Making some noise in data...')
           
    for i, var in enumerate(perturbations):
        data['{}'.format(var)] = coef[i]/data.score + abs(np.random.normal(0,.3,1))
    
    
    for j, var in enumerate(impacts):
        data['{}'.format(var)] = data['{}'.format(var)] * np.random.choice([0,1], 1 , p=[0.5, 1-0.5])

            
    coef = [.01,.011,.002,.04,.035]
    for i, var in enumerate(Perf_parent):
        data['{}'.format(var)] = abs((-coef[i]*data['{}'.format(var)] + coef[i]*data.score + np.random.normal(2,5,1)))
    
    # contraintes sur le max et le min : le min doit tjr etre inférieur au max
    data['max_score_pred2'] = data['max_score_pred']
    data['max_score_pred'] = data[['max_score_pred','min_score_pred']].max(axis=1)
    data['min_score_pred'] = data[['max_score_pred2','min_score_pred']].min(axis=1)
    data = data.drop(columns=['max_score_pred2'])
    
    # ajustement : les valeurs étant trop faibles
    data['max_score_pred'] = data['max_score_pred']*2
    data['mean_score_pred'] = data['mean_score_pred']*2.5 + 0.01*data.score
    data['med_score_pred'] = data['med_score_pred']*2
   
   
    if train==True:
        print('train data and adgency matrix are created successfully. in train data we don\'t add product.')
    else:
        print('test data and adgency matrix are created successfully. in test data we add product P1 and P2 for demostration.')
        data.loc[data.supplier=='F0', 'supplier'] = 'P1'
        data.loc[data.supplier=='F1', 'supplier'] = 'P2'
        data.loc[data.supplier=='F2', 'supplier'] = 'P3'
    
    print('genetrating the adjency matrice for each date...')
    
    
    adjencies_matrix = {}

    for date in data.date.unique():
    
        mats = create_adjency_matrix(graph,data,date,perturbations,train=train)
    
        adjencies_matrix[date] = mats
    
    print('done.')
    
    data = pd.merge(pd.melt(data, id_vars=['date','supplier'], value_vars=perturbations, var_name='perturbations', value_name =                                        'probability' ),
         data.drop(columns=perturbations), 
         on=['date','supplier'])   
    
    if with_score :
        return data, adjencies_matrix
    else:
        return data.drop(columns='score'), adjencies_matrix
    
    
    
    
           
#################################################################################
                 # plot variable importances of a decision tree #
#################################################################################

def plot_importance(model,data,var_exp, algo=''):
    
    '''
    plot les importance des feature pour un random forest 
    model : le model random forest fitté
    var_exp : list des variables explicatives
    ''' 
    print('algo must be Tree or SVC')
    
    if algo == 'Tree':
        
        importances = model.feature_importances_
        indices = np.argsort(importances)

        # Plot the feature importances of the forest
        plt.figure(figsize=(14,8))
        plt.title("Feature importances")
        plt.barh(range(data[var_exp].shape[1]), importances[indices],
               color="r",  align="center")
        # xerr=std[indices],
        plt.yticks(range(data[var_exp].shape[1]), data[var_exp].columns[indices])
        plt.ylim([-1, data[var_exp].shape[1]])
        plt.show()
        
    elif algo == 'SVC':
        
        imp = abs(model.coef_[2])
        imp,names = zip(*sorted(zip(imp,var_exp)))
        plt.figure(figsize=(14,8))
        plt.barh(range(len(names)), imp, align='center')
        plt.yticks(range(len(names)), names)
        plt.show()
    

###############################################################################
                # tranposer un dataframe afin d'avoir les perturbations
        # comme des variables ; ie en colonne
###############################################################################
from itertools import permutations

def reshape_data(data,index,columns,values):
    
    df = pd.merge((data.pivot_table(index=index, columns=columns, values=values).reset_index().rename_axis(None, axis=1)),
                data.drop(columns=[columns,values]).drop_duplicates(),
                on=index)
    return df

    
###############################################################################
     # Pour un noeud donné, mettre à jour les variable à tenir en compte avant de scorer 
###############################################################################

def get_ancesstor(node_data,noeud,graph,perturbations):
	ancesstors = [i for i in list(nx.ancestors(graph, noeud))  if i not in perturbations] + [noeud]
	G = nx.DiGraph()
	for nodes in permutations(ancesstors, r=2):
		G.add_edge(*nodes)
	Edges = [edge for edge in G.edges() if edge in graph.edges()]
	df = node_data[node_data.name.isin(ancesstors)]
	return G, Edges, df

def update_var(model,node_data,noeud,graph,var_exp,perturbations):
    '''
    model : le model de scoring
    node_data : le dataframe contenant les caractéristiques des noeuds (suppliers)
    noeud : le noeud (supplier) qu'on veut scorer
    graph : le graph d'interactions
    var_exp : les variables explicatives à tenir en compte dans la modélisation
    perturbations: la liste des perturbations
    '''
    Predecess = [i for i in list(graph.predecessors(noeud)) if i not in perturbations]
    
    if len(Predecess)==0 :
         pass
    else:
        vals = []
        for predecess in Predecess:
            
            vals.append(score_node(model,node_data,predecess,graph,var_exp,[],perturbations))
            
        node_data.loc[node_data.name==noeud,"mean_score_pred"] = np.mean(vals)
        node_data.loc[node_data.name==noeud,"max_score_pred"] = np.max(vals)
        node_data.loc[node_data.name==noeud,"min_score_pred"] = np.min(vals)
        node_data.loc[node_data.name==noeud,"med_score_pred"] = np.median(vals)
        node_data.loc[node_data.name==noeud,"sd_score_pred"] = np.nanstd(vals)


###############################################################################
                # scorer un fournisseur 
###############################################################################

def score_node(model,node_data,node,graph,var_exp,liste,perturbations):
    
    '''
    model : le model de scoring
    node_data : le dataframe contenant les caractéristiques des noeuds (suppliers)
    node : le noeud (supplier) qu'on veut scorer
    graph : le graph d'interactions
    var_exp : les variables explicatives à tenir en compte dans la modélisation
    liste : une liste pour stocker les scores des parents du noeud en question
    perturbations: la liste des perturbations
    
    output : une valeur dans {1,..,10}
    '''

    Predecessors = [i for i in list(graph.predecessors(node)) if i not in perturbations]
    
    if len(Predecessors)==0:
        
        s_pred = model.predict(node_data.loc[node_data.name==node,var_exp])[0]
        
    else:
        
        for predecessor in Predecessors:
            
            update_var(model,node_data,predecessor,graph,var_exp,perturbations)
            
            liste.append(model.predict(node_data.loc[node_data.name==predecessor,var_exp])[0])
            
            #s_mean = s_pred_list.append(score_node(model,node_data,predecessor, graph,var_to_update,var_exp))
        
        node_data.loc[node_data.name==node,"mean_score_pred"] = np.mean(liste)
        node_data.loc[node_data.name==node,"max_score_pred"] = np.max(liste)
        node_data.loc[node_data.name==node,"min_score_pred"] = np.min(liste)
        node_data.loc[node_data.name==node,"med_score_pred"] = np.median(liste)
        node_data.loc[node_data.name==node,"sd_score_pred"] = np.nanstd(liste)

        s_pred = model.predict(node_data.loc[node_data.name==node,var_exp])[0]
   
    return s_pred

###############################################################################
                # create adjency matrix
###############################################################################

def create_adjency_matrix(tree_graph,data,date,perturbations,train=False):
    '''
    - tree_graph: le graph, les interactions
    - data: les données qui caratérise les noeuds du graph
    - date : la date à la quelle on récupère la matrice d'ajencement
    - perturbations : la liste des perturbations
    - train : si on génére des donnée train (sans les produits)
    
    output:
        une liste dont le premier élément est la matrice d'interact. des fournisseurs et la deuxième est la matrice d'interact. des fournisseurs et des perturbations.
    '''
    sup_sup_adjency = pd.DataFrame(nx.adjacency_matrix(tree_graph).todense())
    
    if train==False:
        supliers = ['P1'] + ['P2'] + ['P3'] +['F'+str(i) for i in range(3,sup_sup_adjency.shape[1])]
    else:
        supliers = ['F'+str(i) for i in range(0,sup_sup_adjency.shape[1])]
    
    sup_sup_adjency.index = supliers
    sup_sup_adjency.columns = supliers

    # interactions perturbationss-fournisseurs
    sup_risk = data.loc[data.date==date, perturbations].T
    sup_risk.columns = supliers

    adjacency_matrix = pd.concat([sup_sup_adjency,sup_risk])

    square_adjacency_matrix = pd.concat([adjacency_matrix,
                                     pd.DataFrame(np.zeros((len(adjacency_matrix),len(perturbations))),columns=perturbations,index=adjacency_matrix.index)]
                                    ,axis=1)
    
    return [sup_sup_adjency , square_adjacency_matrix]


'''
def randomDate (start_date,end_date) :
    time_between_dates = end_date - start_date
    days_between_dates = time_between_dates.days
    random_number_of_days = random.randrange(days_between_dates)
    return start_date + datetime.timedelta(days=random_number_of_days)
'''
