#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 30 22:11:31 2018

@author: shiqi_ning
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 30 18:46:09 2018

@author: shiqi_ning
"""

#Import libraries
import pandas as pd
import numpy as np


#Import plotly
import plotly
plotly.tools.set_credentials_file(username='lemonning', api_key='2VKqv1eNedEuXIqUrKF9')
from plotly import tools
import plotly.plotly as py
import plotly.graph_objs as go
from plotly.grid_objs import Grid, Column
import plotly.figure_factory as ff
from plotly.tools import FigureFactory as FF 
import time
import colorlover as cl




def main():
    
    
    #Import data
    
    myData = pd.read_csv("crimeonlyforweekendnewtype.csv", sep = ',')
    mapData = pd.read_csv("crime_edu_1130.csv", sep = ',')
    zipData_new = pd.read_csv("zipData_new.csv", sep = ',')
    
    
    #Prepare data for visualization
    data = prep_data(myData)
    
    #Draw plots of crime type counts vs day in week
    crime_day(data)
    crime_map(data)
    crime_education(mapData)
    crime_health(data)
    scatterplot(zipData_new)
  
    

def scatterplot(zipData_new):
    
    schoolRateGroup = [1,2,3,4,5]
    
    #Bin school rate group
    zipData_new['new_group'] =  pd.qcut(zipData_new['SchoolRate'], 5, labels = schoolRateGroup)
    
    
    trace = go.Scatter(
        x = zipData_new['new_group'],
        y = zipData_new['CrimeTotal'],
        mode = 'markers'
    )
    
    data = [trace]
    
    # Plot and embed in ipython notebook!
    py.plot(data, filename='Total Crime Type Counts vs Day in a Week_scatter')

    
def prep_data(myData):
    
    #Print data info
    #myData.info()
    
    
    myData = myData[myData.City != 'Kansas City']
    myData = myData.drop(myData.columns[0], axis=1)
    myData = myData.drop(columns=['Other'])

    myData['CrimeTotal'] = myData['Assault'] + myData['Burglary'] + myData['Death']  +\
                           myData['Drug'] + myData['Fraud'] + myData['Robbery'] +\
                           myData['Sexual'] + myData['Theft']
    
    return(myData)





def crime_day(data):
    
    CrimeTypeCount = data.groupby(['Day']).sum().reset_index()
    CrimeTypeCount.head()
   
    # Add bar chart data
    y1 = CrimeTypeCount['Assault']
    y2 = CrimeTypeCount['Burglary'] 
    y3 = CrimeTypeCount['Death'] 
    y4 = CrimeTypeCount['Drug']
    y5 = CrimeTypeCount['Fraud']
    y6 = CrimeTypeCount['Robbery'] 
    y7 = CrimeTypeCount['Sexual'] 
    y8 = CrimeTypeCount['Theft'] 
    
    # Group data together
    #hist_data = [x1, x2, x3, x4, x5, x6]
    
    group_labels = ['Assault', 'Burglary', 'Death', 'Drug', 'Fraud', 'Robbery', 'Sexual', 'Theft']
    
    day_label = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    
    
    # Create distplot with custom bin_size
    #fig = ff.create_distplot(hist_data, group_labels, bin_size=.2)
    
    color_list = cl.scales['8']['div']['RdYlBu']
    
    #Add traces for different crime types for each day of the week
    trace0 = go.Bar(
        x = day_label,
        y = y1,
        name = group_labels[0],
        marker = dict(
            color = color_list[0],
        )
    )
    trace1 = go.Bar(
        x = day_label,
        y = y2,
        name = group_labels[1],
        marker = dict(
            color = color_list[1],
        )
    )
    trace2 = go.Bar(
        x = day_label,
        y = y3,
        name = group_labels[2],
        marker = dict(
            color = color_list[2],
        )
    )
    trace3 = go.Bar(
        x = day_label,
        y = y4,
        name = group_labels[3],
        marker = dict(
            color = color_list[3],
        )
    )
    trace4 = go.Bar(
        x = day_label,
        y = y5,
        name = group_labels[4],
        marker = dict(
            color = color_list[4],
        )
    )
    trace5 = go.Bar(
        x = day_label,
        y = y6,
        name = group_labels[5],
        marker = dict(
            color = color_list[5],
        )
    )
    trace6 = go.Bar(
        x = day_label,
        y = y7,
        name = group_labels[6],
        marker = dict(
            color = color_list[6],
        )
    )
    trace7 = go.Bar(
        x = day_label,
        y = y8,
        name = group_labels[7],
        marker = dict(
            color = color_list[7],
        )
    )
        
    
    #Create subplot layouts
    fig = tools.make_subplots(rows=2, cols=4, subplot_titles=group_labels)
    
    
    #Arrange each subplot
    fig.append_trace(trace0, 1, 1)
    fig.append_trace(trace1, 1, 2)
    fig.append_trace(trace2, 1, 3)
    fig.append_trace(trace3, 1, 4)
    fig.append_trace(trace4, 2, 1)
    fig.append_trace(trace5, 2, 2)
    fig.append_trace(trace6, 2, 3)
    fig.append_trace(trace7, 2, 4)
    
    fig['layout'].update(autosize = True, title='Total Crime Type Counts VS. Day in a Week')
    
    
    #Set x y labels for the subplots
    fig['layout']['xaxis5'].update(title='Day of the Week')
    fig['layout']['xaxis6'].update(title='Day of the Week')
    fig['layout']['xaxis7'].update(title='Day of the Week')
    fig['layout']['xaxis8'].update(title='Day of the Week')

    fig['layout']['yaxis1'].update(title='Number of Crimes')
    fig['layout']['yaxis5'].update(title='Number of Crimes')
        
    

    # Plot!
    py.plot(fig, filename='Total Crime Type Counts vs Day in a Week')
    plotly.offline.plot(fig, filename = 'Total Crime Type Counts vs Day in a Week.html')

    
def crime_map(data):
    
    '''
    city_list = data.City.unique().tolist() 
    
    mapData_new = data.groupby(['City']).sum().reset_index()

    #mapData_new.to_csv("mapData_new.csv", sep = ',', index = False)
    mapData_new.head()

    
    mapData_new['CrimeTotal'] = mapData_new['Assault'] + mapData_new['Burglary'] + mapData_new['Death']  +\
                                mapData_new['Drug'] + mapData_new['Fraud'] + mapData_new['Robbery'] +\
                                mapData_new['Sexual'] + mapData_new['Theft']
    
    
    edu_rank = [8, 110, 7, 33, 14, 83, 26, 98, 118, 29, 101, 42, 5, 3]
    mapData_new['EducationRank'] = edu_rank
    
    city_pop = [983366, 	226505, 687584, 2687682, 719116, 665713, 123400, 4030668, 
                653840, 8580015, 395009, 1573688, 888653, 702756]
    mapData_new['city_pop'] = city_pop
    
    mapData_new['CrimeRate'] = mapData_new['CrimeTotal']/mapData_new['city_pop']*100
    
    #Set hover information

    mapData_new['City'] = mapData_new['City'].replace('NYC', 'New York')
    mapData_new['City'] = mapData_new['City'].replace('LA', 'Los Angeles')    
    mapData_new['City'] = mapData_new['City'].replace('Washington DC', 'Washington')

    
    #City with lat lon
    df = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/2014_us_cities.csv')
    #df.to_csv("df.csv", sep = ',', index = False)
    df.rename(columns={'name':'City'}, inplace=True)    

    df_latlon = pd.read_csv("df_latlon.csv", sep = ',')
    
    map_data_latlon = pd.merge(mapData_new, df_latlon, on="City", how = 'inner').reset_index(drop=True).fillna(0)
    map_data_latlon['text'] = map_data_latlon['City'] + '<br>Crime Rate ' + map_data_latlon['CrimeRate'].astype(str)+'%'
    map_data_latlon['CrimeRate'] = round(map_data_latlon['CrimeRate'], 4)
    '''
    
    
    crime_edu_2017 = pd.read_csv("crime_edu_2017.csv", sep = ',')
    crime_edu_2017['text'] = crime_edu_2017['City'] + '<br>Crime Rate ' + crime_edu_2017['CrimeRate'].astype(str)+'%'

    #Set color list
    color_list = cl.scales['4']['div']['RdYlBu']
    #colors = ["rgb(0,116,217)","rgb(255,65,54)","rgb(133,20,75)","rgb(255,133,27)","lightgreen"]
    
    limits = [(0,10),(10,30),(30,60),(60,100)]

    
    cities = []
    
    for i in range(len(limits)):
        lim = limits[i]
        df_sub = crime_edu_2017[(crime_edu_2017['EducationRank'] >= lim[0]) &
                                 (crime_edu_2017['EducationRank'] <= lim[1])]
        
        city = dict(
            type = 'scattergeo',
            locationmode = 'USA-states',
            lon = df_sub['lon'],
            lat = df_sub['lat'],
            text = df_sub['text'],
            marker = dict(
                size = df_sub['CrimeRate']**4/2,
                # sizeref = 2. * max(df_sub['pop']/scale) / (25 ** 2),
                color = color_list[i],
                line = dict(width=0.5, color='rgb(40,40,40)'),
                sizemode = 'area',
            ),
            name = '{0} - {1}'.format(lim[0],lim[1]) )
        cities.append(city)
        
        
    layout = dict(
            title = '2017 US City Crime Rates vs Education Level',
            showlegend = True,
            geo = dict(
                scope='usa',
                projection=dict( type='albers usa' ),
                showland = True,
                landcolor = 'gainsboro',
                subunitwidth=1,
                countrywidth=1,
                subunitcolor="rgb(255, 255, 255)",
                countrycolor="rgb(255, 255, 255)"
            ),
        )
    
    fig = dict(data=cities, layout=layout)
    py.plot(fig, filename='2017 US City Crime Rates vs Education Rank')

    plotly.offline.plot(fig, filename = '2017 US City Crime Rates vs Education Rank.html')
        
    
def crime_education(mapData):
    
    #zipData_new = mapData.groupby(['ZIP']).sum().reset_index()
    city_list = mapData['City'].unique().tolist()
    zipData_new = mapData
    
    '''
    zipData_new['CrimeTotal'] = zipData_new['Assault'] + zipData_new['Burglary'] + zipData_new['Death']  +\
                                zipData_new['Drug'] + zipData_new['Fraud'] + zipData_new['Robbery'] +\
                                zipData_new['Sexual'] + zipData_new['Theft']
    '''
    
    zipData_new['CrimeTotal']= zipData_new['Acts Causing Harm to Person'] + zipData_new['Controlled Substances'] + \
                               zipData_new['Crimes Leading/Intending to Death'] + zipData_new['Fraud, Deception, or Corruption'] +\
                               zipData_new['Injuries Acts of a Sexual Nature'] + zipData_new['Violence that Involved Property']
    
    
    zipData_new = zipData_new.groupby(['City', 'ZIP']).sum().reset_index()

    zipData_new['SchoolTotal'] = zipData_new['Pri_Sch_Cnt'] + zipData_new['Pub_Sch_Cnt'] + zipData_new['Uni_Cnt']
    
    zipData_new['SchoolPopTotal'] = zipData_new['Pri_Sch_Pop'] + zipData_new['Pub_Sch_Pop'] + zipData_new['Uni_Pop']
    
    
    
    zipData_new['SchoolRate'] =  zipData_new['SchoolTotal']/zipData_new['SchoolPopTotal'] * 100

    max(zipData_new['SchoolRate'] )
    min(zipData_new['SchoolRate'] )
    
    
    schoolRateGroup = ['Top 20%', '20 ~ 40%', '40 ~ 60%', '60 ~ 80%', 'Others']
    
    #Bin school rate group
    zipData_new['SchoolRate_Group'] =  pd.qcut(zipData_new['SchoolRate'], 5, labels = schoolRateGroup)


    
    #BatonRouge = zipData_new[zipData_new['City'] == city_list[0]] 
    #BatonRouge.to_csv('BatonRouge.csv', sep = ',')
    

    
    top20 = zipData_new[zipData_new['SchoolRate_Group'] == 'Top 20%']
    zipData_new.to_csv('zipData_new.csv', sep = ',')
    


    
def crime_health(data):
        
    
    data_new = data.groupby(['City', 'ZIP']).sum().reset_index()

    
    health_city = pd.read_csv("health_cities.csv", sep = ',')
    health_city_2017 = health_city[health_city['year'] == 2017].reset_index()    
    
    #Merge health and crime
    data_new_health = data_new.merge(health_city_2017, how='inner', on = 'City')
    

    
    max(data_new_health['adult_obesity'] )
    min(data_new_health['adult_obesity'] )
    
    max(data_new_health['adult_smoking'] )
    min(data_new_health['adult_smoking'] )
    
    
    adult_obesity_group = ['Lower Obesity Rate', 'Middle Obesity Rate', 'Higher Obesity Rate']
    
    #Bin school rate group
    data_new_health['adult_obesity_group'] =  pd.qcut(data_new_health['adult_obesity'], 3, labels = adult_obesity_group)

    
    
    
    adult_smoking_group = ['Lower Smoking Rate', 'Middle Smoking Rate', 'Higher Smoking Rate']
    
    #Bin school rate group
    data_new_health['adult_smoking_group'] =  pd.qcut(data_new_health['adult_smoking'], 3, labels = adult_smoking_group)
  
    
    
    
    data_new_health.to_csv('data_new_health.csv', sep = ',')

    
main()
