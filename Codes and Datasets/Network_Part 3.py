#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 10 05:50:32 2018

@author: shiqi_ning
"""

import pandas as pd
import numpy as np
import math
import seaborn as sns

import matplotlib.pyplot as plt
from igraph import *
import igraph
import igraph as ig

import plotly
plotly.tools.set_credentials_file(username='lemonning', api_key='2VKqv1eNedEuXIqUrKF9')

import plotly.plotly as py
import plotly.graph_objs as go
from plotly import tools



def main():
    
    #Import dataset
    total_crime = pd.read_csv('crimedata_final_part3.csv', sep = ',')

    #Function that prepare dataframe for network analysis
    network_prep(total_crime)
    
    #Function that run network analysis
    network_analysis()
    indepence_test(total_crime)
    #Make 3d network
    network_3d()
    
def indepence_test(total_crime):
        
    
    #Group all the attributes of interest
    total_crime['Total_Sch_Cnt'] = total_crime['Pri_Sch_Cnt'] + total_crime['Pub_Sch_Cnt'] + total_crime['Uni_Cnt']
    total_crime['Total_Sch_Pop'] = total_crime['Pri_Sch_Pop'] + total_crime['Pub_Sch_Pop'] + total_crime['Uni_Pop']
    total_crime['Total_Sch_Rate'] = total_crime['Total_Sch_Cnt']/total_crime['Total_Sch_Pop']
    
    
    #Merge in health info with crime data
    total_crime['adult_obesity'] = total_crime['City'].map({'Detroit': 0.335, 'Hartford': 0.268, 'Baton Rouge': 0.329, 
                                                   'Denver': 0.164, 'NYC':0.147, 'Chicago': 0.253, 'New Orleans': 0.319,
                                                   'Austin': 0.27, 'LA': 0.214, 'San Francisco': 0.161})
    
    total_crime['adult_smoking'] = total_crime['City'].map({'Detroit': 0.226, 'Hartford': 0.138, 'Baton Rouge': 0.157, 
                                                   'Denver': 0.171, 'NYC':0.119, 'Chicago': 0.146, 'New Orleans': 0.216,
                                                   'Austin': 0.144, 'LA': 0.117, 'San Francisco': 0.099})

    
    #Set bin names for education level
    Total_Sch_Cnt_names = ['Total_Sch_Cnt_1', 'Total_Sch_Cnt_2']
    #Bin Other column into 5 equal groups with [Total_Sch_Cnt_1 < Total_Sch_Cnt_2 < Total_Sch_Cnt_3 < Total_Sch_Cnt_4 < Total_Sch_Cnt_5]
    total_crime['Total_Sch_Lel_Groups'] =  pd.cut(total_crime['Total_Sch_Rate'], 2, labels = Total_Sch_Cnt_names)   
    
    
    #Set bin for health data
    bins_obs=[0, 0.3, 1]
    obs_names = ['obs_1', 'obs_2']
    total_crime['obs_Groups'] =  pd.cut(total_crime['adult_obesity'], bins_obs, labels = obs_names)   
      
    bins_smk=[0, 0.15, 1]
    smk_names = ['smk_1', 'smk_2']
    total_crime['smk_Groups'] =  pd.cut(total_crime['adult_smoking'], bins_smk, labels = smk_names) 
        
    
    #Set bin names for crime
    Theft_names = ['Theft_1', 'Theft_2', 'Theft_3', 'Theft_4', 'Theft_5']
    #Bin Theft column into 5 equal groups with group_1 < group_2 < group_3 < group_4 < group_5
    total_crime['Theft Group'] =  pd.cut(total_crime['Theft'], 5, labels = Theft_names)
        
    Burglary_names = ['Burglary_1', 'Burglary_2', 'Burglary_3', 'Burglary_4', 'Burglary_5']
    #Bin Theft column into 5 equal groups with group_1 < group_2 < group_3 < group_4 < group_5
    total_crime['Burglary Group'] =  pd.cut(total_crime['Burglary'], 5, labels = Burglary_names)
    
    Robbery_names = ['Robbery_1', 'Robbery_2', 'Robbery_3', 'Robbery_4', 'Robbery_5']
    #Bin Theft column into 5 equal groups with group_1 < group_2 < group_3 < group_4 < group_5
    total_crime['Robbery Group'] =  pd.cut(total_crime['Robbery'], 5, labels = Robbery_names)
    
    Drug_names = ['Drug_1', 'Drug_2', 'Drug_3', 'Drug_4', 'Drug_5']
    #Bin Burglar column into 5 equal groups with group_1 < group_2 < group_3 < group_4 < group_5
    total_crime['Drug Group'] =  pd.cut(total_crime['Drug'], 5, labels = Drug_names)
            
    Death_names = ['Death_1', 'Death_2', 'Death_3', 'Death_4', 'Death_5']
    #Bin Theft column into 5 equal groups with group_1 < group_2 < group_3 < group_4 < group_5
    total_crime['Death Group'] =  pd.cut(total_crime['Death'], 5, labels = Death_names)
            
    Fraud_names = ['Fraud_1', 'Fraud_2', 'Fraud_3', 'Fraud_4', 'Fraud_5']
    #Bin Other column into 5 equal groups with group_1 < group_2 < group_3 < group_4 < group_5
    total_crime['Fraud Group'] =  pd.cut(total_crime['Fraud'], 5, labels = Fraud_names)  
        
    Sexual_names = ['Sexual_1', 'Sexual_2', 'Sexual_3', 'Sexual_4', 'Sexual_5']
    #Bin Other column into 5 equal groups with group_1 < group_2 < group_3 < group_4 < group_5
    total_crime['Sexual Group'] =  pd.cut(total_crime['Sexual'], 5, labels = Sexual_names)  
        
    Assault_names = ['Assault_1', 'Assault_2', 'Assault_3', 'Assault_4', 'Assault_5']
    #Bin Other column into 5 equal groups with group_1 < group_2 < group_3 < group_4 < group_5
    total_crime['Assault Group'] =  pd.cut(total_crime['Assault'], 5, labels = Assault_names)  
      
    total_crime.to_csv('total_crime_independence test.csv', header=True,index=False)
    
    #independence test
    #Theft
    type_by_edu = total_crime[['Total_Sch_Lel_Groups', 'Theft Group']]
    type_by_edu.head()
    
    contingency_theft = pd.crosstab(
        type_by_edu['Total_Sch_Lel_Groups'],
        type_by_edu['Theft Group'],
        margins = True
    )
    contingency_theft
    
    #Assigns the frequency values
    sch_count_1 = contingency_theft.iloc[0][0:5].values
    sch_count_2 = contingency_theft.iloc[1][0:5].values
    
    #Plots the bar chart
    fig = plt.figure(figsize=(10, 5))
    sns.set(font_scale=1.8)
    categories = ["Theft_1","Theft_2","Theft_3","Theft_4","Theft_5"]
    p1 = plt.bar(categories, sch_count_2, 0.55, color='#d62728')
    p2 = plt.bar(categories, sch_count_1, 0.55, bottom=sch_count_2)
    plt.legend((p2[0], p1[0]), ('Lower Education Level', 'Higher Education Level'))
    plt.xlabel('Theft Groups')
    plt.ylabel('Count')
    plt.show()
    
    
    f_obs = np.array([contingency_theft.iloc[0][0:5].values,
                  contingency_theft.iloc[1][0:5].values])
    f_obs
    
    from scipy import stats
    stats.chi2_contingency(f_obs)[0:3]
    
    
    
def network_prep(total_crime):
    
    #Select columns
    crime = total_crime[['Theft', 'Burglary', 'Robbery', 'Drug', 'Fraud', 'Assault', 'Death', 'Sexual']]
    
    #Find the two most frequent crime types
    maxtype = pd.DataFrame(np.sort(crime)[:,-2:], columns=['Max2','Max'])
    
    #Merge datasets to further print the corresponding column name(crime type)
    df = pd.concat([crime, maxtype], axis=1)
    
    
    crime_new = df[['Theft', 'Burglary', 'Robbery', 'Drug', 'Fraud', 'Assault', 'Death', 'Sexual', 'Max2']]
    crime_ntwk = df[['Theft', 'Burglary', 'Robbery', 'Drug', 'Fraud', 'Assault', 'Death', 'Sexual']]
    
    #Change the relative values into crime type
    crime_new['Max2'] = crime_new.T.apply(lambda x: x.nlargest(2).idxmin())
    #Find the max element in the dataframe and print the corresponding column name
    crime_ntwk['Max'] = crime_ntwk.idxmax(axis=1)
    
    #Merge data and count the frequency, and set column name
    DataFrame = pd.concat([crime_ntwk['Max'], crime_new['Max2']], axis=1)
    network_crime = DataFrame.groupby(['Max', 'Max2']).size().reset_index(name="Weight")
    
    #Drop rows with same value
    network_crime = network_crime[network_crime['Max'] != network_crime['Max2']]
    
    #Export the data to txt file
    network_crime.to_csv('network_crime_data.txt', header=None,index=False, sep=' ', mode='w')
    

    

def network_analysis():
    
    #Open data and read in by igraph
    with open('network_crime_data.txt', 'r', encoding = 'utf-8') as input_file:
        network = igraph.Graph.Read_Ncol(input_file, names=True, weights = True)
        print(input_file)
    
    
    
    #________Draw the network graph_____________________
    #Set the weight for each row by proportion, so that I can get better visuals for edge widths
    weight = []
    weight_color = []
    for i in range(len(network.es['weight'])):
        weight.append(network.es['weight'][i])
        weight_color.append(network.es['weight'][i]**0.5/20)
    
    
    #Set the size for each vertex and vertex label based on degree,
    #so the vertex and labels with higher degree can have bigger sized nodes and labels
    size = []
    for i in range(len(network.degree())):
        size.append(network.degree()[i]*5)
    
    
    #Set visual commands
    vis_style = {}
    vis_style["vertex_label"] = network.vs["name"]
    vis_style["vertex_label_size"] = size
    vis_style["vertex_label_color"] = 'black'
    vis_style["vertex_color"] = ['red', 'blue', 'blue', 'blue', 'blue', 'blue', 'blue', 'red']
    vis_style["vertex_size"] = size
    vis_style["edge_color"] = 'black'
    vis_style["edge_label"] = weight
    vis_style["edge_width"] = weight_color
    vis_style["margin"] = 200
    vis_style["bbox"] = (1800, 1800)
    
    
    
    #Change nodes color based on degree
    #Check the frequency of each elements in the degree list
    dict((x,network.degree().count(x)) for x in set(network.degree()))
    
    '''
    #Find the comminities in the networks
    i = network.community_infomap()
    #This will give me 14 colors
    pal = igraph.drawing.colors.ClusterColoringPalette(len(i))
    vis_style['vertex_color'] = pal.get_many(i.membership)
    '''
    
    
    #Set Layout format
    layout = network.layout("kk")
    pl1 = igraph.plot(network,layout = layout, **vis_style)
    pl1.show()



    
        
    #________Summarize the results_____________________
    
    print("\nFor igraph-python:\n")
    print("Crime Dataset:")
    print("Nodes Labels:", network.vs["name"])
    print("Nodes:", len(igraph.VertexSeq(network)), ", Edges:", len(igraph.EdgeSeq(network)))
    print("Density:", round(network.density(), 6), "\n")
    print("Degree:", network.degree(), "\n")
    print("Average degree of the neighbors for each vertex:", network.knn(vids=None, weights=weight)[0], "\n")
    print("Eigenvector Centralities of the Vertices:", network.eigenvector_centrality(weights=weight), "\n")
    print("Betweenness:", network.betweenness(weights=weight), "\n")
    print("The local transitivity (clustering coefficient):", network.transitivity_local_undirected(vertices=None, mode="nan", weights=weight), "\n")
    
    
    

    

def network_3d():


    #Read in dataset by using igraph
    with open('network_crime_data.txt', 'r', encoding = 'utf-8') as input_file:
        network = igraph.Graph.Read_Ncol(input_file, names=True, weights = True)
        print(network)
    
    #Set weight and size
    weight = []
    weight_color = []
    for i in range(len(network.es['weight'])):
        weight.append(network.es['weight'][i])
        weight_color.append(network.es['weight'][i]/50)
    
    
    #Set the size for each vertex and vertex label based on degree,
    #so the vertex and labels with higher degree can have bigger sized nodes and labels
    size = []
    for i in range(len(network.degree())):
        size.append(network.degree()[i]*2)
    
    
    #Set layout
    layt = network.layout('kk', dim=3) 
        
    # Count total number of vertices
    N = len(ig.VertexSeq(network))
        
    # Setup 3d nodes position
    Xn=[layt[k][0] for k in range(N)]# x-coordinates of nodes
    Yn=[layt[k][1] for k in range(N)]# y-coordinates
    Zn=[layt[k][2] for k in range(N)]# z-coordinates
        
    
    # Initialize list to collect edges position
    Xe=[]
    Ye=[]
    Ze=[]
      
    
    # Initialize edge width and color lists
    edge_width = []    
    # Iterate through all edges to set positions, widths, and colors
    for edge in network.es:
        start = edge.source
        end = edge.target
            
        Xe+=[layt[ start ][0],layt[ end ][0], None]# x-coordinates of edge ends
        Ye+=[layt[ start ][1],layt[ end ][1], None]  
        Ze+=[layt[ start ][2],layt[ end ][2], None]  
            
        edge_width.append(edge['weight'])
        
      
    #Set node label format
    node_label = []
    # Iterate through all edges to set positions, widths, and colors
    for i in range(len(network.vs)):
    
        
        # Set labels to display interactively
        label = ''.join([network.vs['name'][i], '<br>',
                         'Degree:', 
                         str(network.degree()[i])])
            
    
        # Append label, size, and color to corresponding lists
        node_label.append(label)
    
        
    # Set trace of edges
    trace1=go.Scatter3d(x=Xe,
                        y=Ye,
                        z=Ze,
                        mode='lines',
                        line=dict(color = weight_color,
                                  cmin = min(weight_color),
                                  cmax = max(weight_color),
                                  colorscale='Reds', 
                                  width=1),
                        hoverinfo='none',
                        
                        )
        
    # Set trace of nodes
    trace2=go.Scatter3d(x=Xn,
                        y=Yn,
                        z=Zn,
                        mode='markers',
                        name='actors',
                        marker=dict(symbol='circle',
                                    size=size,
                                    color = size,
                                    cmin = min(size),
                                    cmax = max(size),
                                    colorscale='Reds',
                                    line=dict(color='rgb(50,50,50)', width=0.5)
                                    ),
                        text=node_label,    
                        hoverinfo='text'
                        )
                            
    # Setup axis dictionary (mute the axes)
    axis=dict(showbackground=False,
              showline=False,
              zeroline=False,
              showgrid=False,
              showticklabels=False,
              title='',
              )
    
    # Setup the layout
    layout = go.Layout(
            title="Network of Crime Types (3D visualization)",
            width=600,
            height=600,
            showlegend=False,
            scene=dict(
                    xaxis=dict(axis),
                    yaxis=dict(axis),
                    zaxis=dict(axis),
                    ),
                    margin=dict(
                            t=100
                            ),
                    hovermode='closest',
                    )
        
    # Final combination of data to plot
    data=[trace1, trace2]
    fig=go.Figure(data=data, layout=layout)
    
        
    # 3D network plotting
    plotly.offline.plot(fig, validate=False, filename='3d_network.html')
    #py.plot(fig, validate=False, filename='3d_network.html')

    

main()










