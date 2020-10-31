# Libraries
from sklearn import datasets
import pandas as pd
import numpy as np
from sklearn import metrics
from pandas.tools.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import cross_validation
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn import preprocessing
from sklearn.decomposition import PCA
#import xgboost as xgb
from sklearn.preprocessing import label_binarize
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import random
from sklearn import preprocessing
##input crime data contians 10 cities crime records



##create function model selection for select optimal model
def ModelSelection(X_train,  Y_train):

   # X_train, X_validate, Y_train, Y_validate = cross_validation.train_test_split(X, Y, test_size=test_size, random_state=seed)

    # Setup 10-fold cross validation to estimate the accuracy of different models
    # Split data into 10 parts
    # Test options and evaluation metric
    
    num_instances = len(X_train)
   
    scoring = 'accuracy'

######################################################
# Use different algorithms to build models
######################################################

# Add each algorithm and its name to the model array
    models = []
    models.append(('KNN', KNeighborsClassifier()))
    models.append(('CART', DecisionTreeClassifier()))
    models.append(('NB', GaussianNB()))
    models.append(('SVM', SVC()))
    #models.append(('RF',  RandomForestRegressor(n_estimators=20, random_state=0) ))

    # Evaluate each model, add results to a results array,
    # Print the accuracy results (remember these are averages and std
    results = []
    names = []
    resultsmean=[]
    for name, model in models:
        if name=='RF':
            
            #Y_train=onehot(Y_train)
            print (Y_train.unique())

        kfold = cross_validation.KFold(n=num_instances, n_folds=10, random_state=7)
        cv_results = cross_validation.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
        print (cv_results)
        results.append(cv_results)
        resultsmean.append(cv_results.mean())
        names.append(name)
        msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
        print(msg)
        #if name=="CART":
            #false_positive_rate, true_positive_rate, thresholds = roc_curve(actual, predictions)
            #roc_auc = auc(false_positive_rate, true_positive_rate)

        
    ##print out the best model and return it for final prediction 
    best_model=models[resultsmean.index(max(resultsmean))]
    print("\n")
    print("the best model for this case is %s " %best_model[0])
    print("\n")
    return best_model[1]


def main():
    typestr='Theft'

# import data
    total_crime=pd.read_csv("crimedata_final_part3.csv")
    total_crime['Total_Sch_Cnt'] = total_crime['Pri_Sch_Cnt'] + total_crime['Pub_Sch_Cnt'] + total_crime['Uni_Cnt']
    total_crime['Total_Sch_Pop'] = total_crime['Pri_Sch_Pop'] + total_crime['Pub_Sch_Pop'] + total_crime['Uni_Pop']
    total_crime['Total_Sch_Rate'] = total_crime['Total_Sch_Cnt']/total_crime['Total_Sch_Pop']
    data=total_crime
    rainy = [] 
    rainy_weather = frozenset(["Moderate or heavy rain shower", "Mist", "Light rain", 
                         'Light drizzle','Light snow', 'Moderate snow',
                         'Heavy snow','Light freezing rain', 'Fog', 
                         'Light snow showers','Light rain shower', 
                         'Moderate or heavy rain with thunder','Moderate rain',
                         'Blizzard','Light sleet', 'Moderate or heavy snow showers',
                         'Moderate or heavy sleet', 'Patchy light rain with thunder',
                         'Heavy rain', 'Patchy light drizzle', 'Patchy light rain',
                         'Torrential rain shower', 'Moderate rain at times', 'Freezing fog',
                         'Patchy heavy snow', 'Patchy light snow with thunder',
                         'Blowing snow', 'Ice pellets', 'Patchy light snow',
                         'Light sleet showers', 'Heavy rain at times'])
    for i in range(data.shape[0]):
        if data.Weather[i] in rainy_weather:
            rainy.append("Rainy")
        else:
            rainy.append("Not rainy")
    data["Rainy"] = rainy
    #data=data[['Day','Month','Moon_Phase','MaxTemperature','Moon_Illumination','Rainy','Total_Sch_Rate','max']]

    

    data['robbery']=data['max']
    data['robbery'][data[typestr] >data[typestr].median()] = "1"
    data['robbery'][data[typestr] <=data[typestr].median()] = "0"
    data=data[['Day','Month','Moon_Phase','MaxTemperature','Moon_Illumination','Rainy','Total_Sch_Rate','max','robbery']]

    data['Moon_Phase']=data['Moon_Phase'].replace(data['Moon_Phase'].unique(), ["0","1","2","3","4","5","6","7"])
    data['Rainy']=data['Rainy'].replace(data['Rainy'].unique(), ["0","1"])


    myData = data
    #myData=myData.drop(['Unnamed: 0'],axis=1)
    #column=['Assault','Burglar','Other','Theft']
    #myData=prepanalysis(myData,column)
    myData=myData.dropna()
    ##summarize the data and show the plot
   # Summarizedata(myData)
    #if(myData.columns.contains('Moon_Phase')):
        #myData= get_dummy(myData)
    valueArray = myData.values
    
    
   
 ######################################################
# ##RUN _A with all features
######################################################

    X = valueArray[:,0:7]

##normalize data 
    X=preprocessing.normalize(X,axis=0)

    Y = valueArray[:,8]
    test_size = 0.20
    seed = 7
    num_folds=10
##split data wiht train and test
    X_train, X_validate, Y_train, Y_validate = cross_validation.train_test_split(X, Y, test_size=test_size, random_state=seed)
    
##select the best model with highest accurancy and get itfor testing set
    model=ModelSelection(X_train, Y_train)

##test the best model
    #nb=GaussianNB()
    
    #final_prediction(model, X_train,Y_train,X_validate)
    model.fit(X_train, Y_train)
    predictions = model.predict(X_validate)
    print()
    print(accuracy_score(Y_validate, predictions))
    print(confusion_matrix(Y_validate, predictions))
    print(classification_report(Y_validate, predictions))
    probs = model.predict_proba(X_validate)
    
    preds = probs[:,1]
    preds= pd.to_numeric(preds, errors='coerce')
    Y_validate=pd.to_numeric(Y_validate, errors='coerce')
    fpr, tpr, threshold = metrics.roc_curve(Y_validate, preds)
    roc_auc = metrics.auc(fpr, tpr)
    plt.title('Receiver Operating Characteristic for predict '+ typestr)
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()
    
if __name__ == "__main__":
    main()  