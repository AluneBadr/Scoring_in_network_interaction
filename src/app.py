import streamlit as st
import os
import pickle
from PIL import Image
import utils

st.set_page_config(layout="wide")
#import partition


exec(open("./src/library.py").read())


perturbations = ['TRP', 'GEO', 'GRV','CR', 'AUTR','FIN', 'MP','LKS', 'ADM','INC']
impacts = ['qlty_impact','on_time_impact','qtty_impact']
Perf_parents = ["sd_score_pred","max_score_pred","mean_score_pred","med_score_pred","min_score_pred"]


def loadData():
    
    
    image = Image.open('./image/Picture1.png')
    donnee = Image.open('./image/data.png')
    
    #path="C:/Users/alioka/Stage/DILA/Project/Classsification"
    #os.chdir(path)
    # model
    with open("./output/model_tree.pkl", "rb") as f:
        Tree = pickle.load(f)
        
    with open("./output/model_lag1_tree.pkl", "rb") as f:
        Tree_lag1 = pickle.load(f)
        
    with open("./output/model_lag2_tree.pkl", "rb") as f:
        Tree_lag2 = pickle.load(f)
    
    with open("./output/model_SVC.pkl", "rb") as f:
        SVC = pickle.load(f)
    # train data
    data = pd.read_csv('./data/data_generate.csv')
    #reshape 
    #d = min(data.date)
        
    # train data
    train = pd.read_csv('./data/train_sample.csv')
    return image,donnee,Tree,Tree_lag1,Tree_lag2, SVC , data, train

# plotting groups using pca.  
st.set_option('deprecation.showPyplotGlobalUse', False)


###########################################################################
#Functions for streamlit
##########################################################################

def plot_groupes(data):

    pca = PCA(n_components=3)
    var=  [i for i in data.columns if i not in ['Unnamed: 0','supplier','date','score','cluster']]
    # fit X and apply the reduction to X 
    x_3d = pca.fit_transform(data[var])
    # Set a 3 KMeans clustering
    kmeans = KMeans(n_clusters=10, random_state=0)
    #cluster = AgglomerativeClustering(n_clusters=10, affinity='euclidean', linkage='ward')
    # Compute cluster centers and predict cluster indices
    X_clustered = kmeans.fit_predict(x_3d)
    #X_clustered = cluster.fit_predict(x_3d)

    LABEL_COLOR_MAP = {0 : 'groupe0',1 : 'groupe1',2 : 'groupe2',
                           3 : 'groupe3',4 : 'groupe4',5 : 'groupe5',
                           6 : 'groupe6',7 : 'groupe7',8 : 'groupe8',9 : 'groupe9'
                           }
    label_color = [LABEL_COLOR_MAP[l] for l in X_clustered]
    fig = px.scatter(x=x_3d[:,0], y=x_3d[:,1], color=label_color)
    fig.update_layout(
    title="Les différents groupes de fournisseurs",
    xaxis_title="premier axe principal",
    yaxis_title="deuxième axe principal",
    legend_title="Groupes")
    st.plotly_chart(fig)

def pie_chart(data):
    df = pd.DataFrame(pd.value_counts(data.score))
    fig = go.Figure(data=[go.Pie(labels=df.index, values=df.score, hole=.3)])
    fig.update_layout(
        title="Proportion de fournissuers par score de performance",
        legend_title="Score de performance")
    st.plotly_chart(fig)
    
def plot_cor_matrice(data):
    X_train = data[var_x]
    Y_train = data['score']
    plot_correlation(pd.concat([X_train,Y_train], axis=1))
    

def plot_network(matrices,date):
    t = date
    Net = nx.from_pandas_adjacency(matrices[t][1] , create_using=nx.DiGraph)

    color_map = []
    for node in Net:
        if node in perturbations:
            color_map.append('#FF0000')
        elif node in ['P1', 'P2','P3']:
            color_map.append('yellow')
        else: 
            color_map.append('blue')  
            #color_map.append('#CCFFAA') 

    
    #, node_color=color_map
    #pos = spectral_layout(Net)
    #pos=nx.kamada_kawai_layout(Net)
    edges = [edge for edge in Net.edges()]
    pos = nx.spring_layout(Net,k=10*1/np.sqrt(len(Net.nodes())), iterations=20)
    fig = plt.figure(figsize=(15, 12))
    nx.draw_networkx_nodes(Net, pos, cmap=plt.get_cmap('jet'), node_color = color_map, node_size = 1000)
    nx.draw_networkx_labels(Net, pos,font_size = 20)
    nx.draw_networkx_edges(Net, pos, edgelist=edges, arrowsize=25,width=.5, arrows=True)
    #fig = nx.draw(Net,pos=pos,node_color=color_map, with_labels=True,arrows=True,node_size=1000)
    st.pyplot(fig)
    return Net
    
def plot_importance(model,df,var_exp, algo=''):
    
    '''
    plot les importance des feature pour un random forest 
    model : le model random forest fitté
    var_exp : list des variables explicatives
    '''
    fig, ax = plt.subplots(1,2 ,figsize=(17, 8))
    X_train = df[var_exp]
    Y_train = df['score']
    mask = np.zeros_like(pd.concat([X_train,Y_train], axis=1).corr())
    mask[np.triu_indices_from(mask)] = 1
    sns.heatmap(pd.concat([X_train,Y_train], axis=1).corr(), mask= mask, ax= ax[1], annot= False)
    ax[1].set_title('Correlation map')
    
    if algo == 'Tree':
        
        importances = model.feature_importances_
        indices = np.argsort(importances)

        # Plot the feature importances of the forest
        #plt.figure(figsize=(14,8))
        
        ax[0].barh(range(df[var_exp].shape[1]), importances[indices],
               color="b",  align="center")
        ax[0].set_title("Feature importances")
        # xerr=std[indices],
        ax[0].set_yticks(range(df[var_exp].shape[1]))
        ax[0].set_yticklabels(df[var_exp].columns[indices])
        ax[0].set_ylim([-1, df[var_exp].shape[1]])
        
        
    elif algo == 'Svc':
        
        imp = abs(model.coef_[2])
        imp,names = zip(*sorted(zip(imp,var_exp)))
        ax[0].barh(range(len(names)), imp, align='center')
        ax[0].set_title("Feature importances")
        ax[0].set_yticks(range(len(names)))
        ax[0].set_yticklabels(names)
        
        
    else:
        print('algo must be Tree or Svc')
        
    st.pyplot(fig)


def generate_test_data(parent_input,level_input,periods_input):
    
    New_tree = utils.create_tree_graph(parent_input,level_input,remove_first=False) # un graph de 3 niveaux. chaque neud a 3 prédécesseurs
    # add / remoove somme link
    New_tree.add_edge('F16', 'F0')
    New_tree.add_edge('F36', 'F0')
    New_tree.add_edge('F19','F0')
    New_tree.add_edge('F15', 'F2')
    New_tree.add_edge('F26', 'F2')
    New_tree.add_edge('F39', 'F1')
    New_tree.add_edge('F29', 'F1')
    New_tree.remove_edge('F1','F0')
    New_tree.remove_edge('F2','F0')
    
    #nb_supliers = len(New_tree.nodes)
    
    start_date = datetime.date(2020, 8, 1)

    New_data, new_matrices = utils.generate_data(New_tree,periods_input, start_date,perturbations,impacts,Perf_parents,centers=None, with_score = True, train=False)

    New_data['name']= New_data.supplier

    New_data_reshaped = utils.reshape_data(New_data,["date","supplier"],columns="perturbations",values="probability")
    
    # return data in wide format and adjency matrice in diffrent date
    return New_data_reshaped,new_matrices

import datetime
step = 8
def previous_date(p_date, step=8) :
    p_date =p_date - datetime.timedelta(days=step)
    return p_date


def st_plot_net(Net,edges, edg_lab, son):
    fig = plt.figure(figsize=(12, 9))
    color_map = []
    for node in Net:
        if node == son:
            color_map.append('#82E0AA')
        else: 
            color_map.append('blue')  
    pos = nx.spring_layout(Net,k=10*1/np.sqrt(len(Net.nodes())), iterations=20)
    nx.draw_networkx_nodes(Net, pos, cmap=plt.get_cmap('jet'), node_size = 1000,node_color = color_map)
    nx.draw_networkx_labels(Net, pos,font_size = 20)
    nx.draw_networkx_edges(Net, pos, edgelist=edges,label = edg_lab, arrowsize=30,width=.5, arrows=True)
    #fig = nx.draw(Net,pos=pos,node_color=color_map, with_labels=True,arrows=True,node_size=1000)
    return(st.pyplot(fig))

#####################################################################################"
# End Functions
####################################################################################

### slide bar
st.sidebar.title("Variable to generate data : ") 
periods_input = st.sidebar.slider('How many periods would you like to generate?',min_value = 10, max_value = 100)
level_input = st.sidebar.number_input( 'How many levels of suppliers would you like to generate?', 3, 6, 3)
parent_input = st.sidebar.number_input('How many node\'s parent would you like to generate?',3, 6, 3)

## load data
image, donnee,Tree,Tree_lag1,Tree_lag2, SVC , data , train= loadData()
var_x = [i for i in train.columns if i not in ['Unnamed: 0','supplier','date','score','cluster',"score_T-1","score_T-2"]]

# creation des lags
date = data.date.unique().tolist()
d = date[9] # date choisi pour l'entrainement
d = datetime.datetime.strptime(str(d),"%Y-%m-%d").date()

lag1 = utils.reshape_data(data[data.date==str(previous_date(d,1*step))],["date","supplier"],\
                    columns="perturbations",values="probability")['score']

lag2 = utils.reshape_data(data[data.date==str(previous_date(d,1*step))],["date","supplier"],\
                    columns="perturbations",values="probability")['score']

train["score_T-1"] = lag1+0.3*train['score']
train["score_T-2"] = lag2+0.1*train['score']

##### main()
#@st.cache(allow_output_mutation=True)
def main():
    st.title("Scoring in a network interaction : An Example in Supply Chain")
    st.markdown('---')
    st.image(image, caption='An Example of suppliers interaction', use_column_width=True)

    st.title("data")

    st.image(donnee, caption='Description of variables', use_column_width=True)
    #st.title('Hypothèses')

    st.title("EDA")
    value = st.selectbox("Choose a date", data.date.unique().tolist())
    df = utils.reshape_data(data[data.date==value],["date","supplier"],columns="perturbations",values="probability")
    plot_groupes(df)
    pie_chart(df)
    #if(st.checkbox("see dendrogram")):
        #st.image(dendrom, caption='Le dedrogram', use_column_width=True)
    
    st.title("Selection")
    Test_data, Matrices = generate_test_data(parent_input,level_input,periods_input)
    Test_data['score_T-1'] = 0
    Test_data['score_T-2'] = 0
    
    st.sidebar.title("Scoring : ")     
        
    with st.beta_expander('Choose  a product in the sidebar to see its best supplier', True):
    
        choose_model = st.sidebar.selectbox("Choose the ML Model",
        ["Decision Tree", "Support Vector Classifier"])
        
        choose_product = st.sidebar.selectbox("Choose the Product",
        ["P1", "P2","P3"])
       
        
        #dt = st.sidebar.selectbox("Choose the date", Test_data.date.unique().tolist()[2:])
    
        if(choose_model == "Decision Tree"):
            Lags = st.sidebar.selectbox("Include Lags (the past)",  [0, 1,2])
            if Lags==0:
                dt = st.sidebar.selectbox("Choose a date", Test_data.date.unique().tolist()[::-1])
                Net = plot_network(Matrices,dt)
                
                plot_importance(Tree,train,var_x, algo='Tree')
                df_p = Test_data[Test_data.date==dt][Test_data.supplier.isin(list(Net.predecessors(choose_product)))]

                P_suplier_Score = []
                df_t = Test_data[Test_data.date==dt]
                
                df_t = Test_data[Test_data.date==dt]
                df_t['name'] = df_t.supplier
                #df_t.loc[df_t.suplier=='F1',['qlty_score','qtty_score','one_time_score']]=3*[7]

                Supliers = [i for i in list(Net.predecessors(choose_product)) if i not in perturbations]

                for sup in Supliers:

                    P_suplier_Score.append(utils.score_node(Tree,df_t,sup,Net,var_x,[],perturbations))
                    
                Scores = pd.DataFrame.from_dict({'Supliers':Supliers, 'P_suplier_Score':P_suplier_Score})
                Scores = Scores.sort_values('P_suplier_Score', ascending = True) 
                f = go.Figure(go.Bar(x=Scores.P_suplier_Score,y=Scores.Supliers, name=choose_product,orientation='h',marker=dict(
                color='rgba(50, 171, 96, 0.6)',line=dict(color='rgba(50, 171, 96, 1.0)',width=1) )))
                f['layout'].update(showlegend=True,title='Scores des Fournisseurs de {} au {}'.format(choose_product,dt))
                st.plotly_chart(f)
				
                if st.checkbox('Display Score driver of suppliers  for selected product '):
                    #df['mean_risque'] = df[perturbations].mean(axis = 1)
                    cols = ['date','supplier']+var_x
                    st.write(df_p[cols])
                sup =  st.selectbox("Choose a Supplier to see its ancesstors",  list(df_p.name.values))
                Net = nx.from_pandas_adjacency(Matrices[dt][1] , create_using=nx.DiGraph)
                G, Edges, df = utils.get_ancesstor(df_t,sup,Net,perturbations)
                st_plot_net(G,Edges, [str(i) for i in list(df.score.values)],sup)
                st.write(df[["name","score"]])

            elif Lags ==1:
                dt = st.sidebar.selectbox("Choose a date", Test_data.date.unique().tolist()[::-1][:-1])
                Net = plot_network(Matrices,dt)
                Test_data.loc[Test_data.date==dt,'score_T-1'] = Test_data[Test_data.date==previous_date(dt,1*step)]['score'].tolist()+ \
                np.random.normal(2,0.1, len(Test_data[Test_data.date==dt]))

                var = var_x+['score_T-1']
                
                plot_importance(Tree_lag1,train,var, algo='Tree')
                
                df_p = Test_data[Test_data.date==dt][Test_data.supplier.isin(list(Net.predecessors(choose_product)))]

                P_suplier_Score = []
                Net = nx.from_pandas_adjacency(Matrices[dt][1] , create_using=nx.DiGraph)
                df_t = Test_data[Test_data.date==dt]
                df_t['name'] = df_t.supplier
                #df_t.loc[df_t.suplier=='F1',['qlty_score','qtty_score','one_time_score']]=3*[7]

                Supliers = [i for i in list(Net.predecessors(choose_product)) if i not in perturbations]

                for sup in Supliers:
                    P_suplier_Score.append(utils.score_node(Tree_lag1,df_t,sup,Net,var,[],perturbations))
                    
                Scores = pd.DataFrame.from_dict({'Supliers':Supliers, 'P_suplier_Score':P_suplier_Score})
                Scores = Scores.sort_values('P_suplier_Score', ascending = True) 
                f = go.Figure(go.Bar(x=Scores.P_suplier_Score,y=Scores.Supliers, name=choose_product,orientation='h',marker=dict(
                 color='rgba(50, 171, 96, 0.6)',line=dict(color='rgba(50, 171, 96, 1.0)',width=1) )))
                f['layout'].update(showlegend=True,title='Scores des Fournisseurs de {} au {}'.format(choose_product,dt))
                st.plotly_chart(f)

                if st.checkbox('Display Score driver  of suppliers for selected product '):
                    #df['mean_risque'] = df[perturbations].mean(axis = 1)
                    cols = ['date','supplier']+var
                    st.write(df_p[cols])
					
                sup =  st.selectbox("Choose a Supplier to see its ancesstors",  list(df_p.name.values))
                Net = nx.from_pandas_adjacency(Matrices[dt][1] , create_using=nx.DiGraph)
                G, Edges, df = utils.get_ancesstor(df_t,sup,Net,perturbations)
                st_plot_net(G,Edges, [str(i) for i in list(df.score.values)],sup)
                st.write(df[["name","score"]])

            elif Lags ==2:
                dt = st.sidebar.selectbox("Choose a date", Test_data.date.unique().tolist()[::-1][:-2])
                Net = plot_network(Matrices,dt)
                Test_data.loc[Test_data.date==dt,'score_T-1'] = Test_data[Test_data.date==previous_date(dt,1*step)]['score'].tolist()+ \
                np.random.normal(2,0.1, len(Test_data[Test_data.date==dt]))
                Test_data.loc[Test_data.date==dt,'score_T-2'] = Test_data[Test_data.date==previous_date(dt,2*step)]['score'].tolist()

                var = var_x+['score_T-1','score_T-2']
                
                plot_importance(Tree_lag2,train,var, algo='Tree')
                
                df_p = Test_data[Test_data.date==dt][Test_data.supplier.isin(list(Net.predecessors(choose_product)))]

                P_suplier_Score = []
                Net = nx.from_pandas_adjacency(Matrices[dt][1] , create_using=nx.DiGraph)
                df_t = Test_data[Test_data.date==dt]
                df_t['name'] = df_t.supplier
                #df_t.loc[df_t.suplier=='F1',['qlty_score','qtty_score','one_time_score']]=3*[7]

                Supliers = [i for i in list(Net.predecessors(choose_product)) if i not in perturbations]

                for sup in Supliers:
                    P_suplier_Score.append(utils.score_node(Tree_lag2,df_t,sup,Net,var,[],perturbations))
                    
                Scores = pd.DataFrame.from_dict({'Supliers':Supliers, 'P_suplier_Score':P_suplier_Score})
                Scores = Scores.sort_values('P_suplier_Score', ascending = True) 
                f = go.Figure(go.Bar(x=Scores.P_suplier_Score,y=Scores.Supliers, name=choose_product,orientation='h',marker=dict(
                color='rgba(50, 171, 96, 0.6)',line=dict(color='rgba(50, 171, 96, 1.0)',width=1) )))
                f['layout'].update(showlegend=True,title='Scores des Fournisseurs de {} au {}'.format(choose_product,dt))
                st.plotly_chart(f)  

                if st.checkbox('Display Score driver of suppliers for selected product '):
                    #df['mean_risque'] = df[perturbations].mean(axis = 1)
                    cols = ['date','supplier']+var
                    st.write(df_p[cols])

                sup =  st.selectbox("Choose a Supplier to see its ancesstors",  list(df_p.name.values))
                Net = nx.from_pandas_adjacency(Matrices[dt][1] , create_using=nx.DiGraph)
                G, Edges, df = utils.get_ancesstor(df_t,sup,Net,perturbations)
                st_plot_net(G,Edges, [str(i) for i in list(df.score.values)],sup)
                st.write(df[["name","score"]])
				
        if(choose_model == "Support Vector Classifier"):
            dt = st.sidebar.selectbox("Choose a date", Test_data.date.unique().tolist()[::-1])
            Net = plot_network(Matrices,dt)

            plot_importance(SVC,train,var_x, algo='Svc')
            df_p = Test_data[Test_data.date==dt][Test_data.supplier.isin(list(Net.predecessors(choose_product)))]

                
            P_suplier_Score = []
            Net = nx.from_pandas_adjacency(Matrices[dt][1] , create_using=nx.DiGraph)
            df_t = Test_data[Test_data.date==dt]
            df_t['name'] = df_t.supplier
            #df_t.loc[df_t.suplier=='F1',['qlty_score','qtty_score','one_time_score']]=3*[7]

            Supliers = [i for i in list(Net.predecessors(choose_product)) if i not in perturbations]

            for sup in Supliers:
                P_suplier_Score.append(utils.score_node(SVC,df_t,sup,Net,var_x,[],perturbations))
                        
            f = go.Figure(go.Bar(x=P_suplier_Score,y=Supliers, name=choose_product,orientation='h',marker=dict(
        color='rgba(50, 171, 96, 0.6)',line=dict(color='rgba(50, 171, 96, 1.0)',width=1) )))
            f['layout'].update(showlegend=True,title='Scores des Fournisseurs de {} au {}'.format(choose_product,dt))

            st.plotly_chart(f)                
             
            if st.checkbox('Display Score driver of suppliers  for selected product '):
                    #df['mean_risque'] = df[perturbations].mean(axis = 1)
                    cols = ['date','supplier']+var_x
                    st.write(df_p[cols])
					
                sup =  st.selectbox("Choose a Supplier to see its ancesstors",  list(df_p.name.values))
                Net = nx.from_pandas_adjacency(Matrices[dt][1] , create_using=nx.DiGraph)
                G, Edges, df = utils.get_ancesstor(df_t,sup,Net,perturbations)
                st_plot_net(G,Edges, [str(i) for i in list(df.score.values)],sup)
                st.write(df[["name","score"]])  
				
				
if __name__ == "__main__":
	main()
