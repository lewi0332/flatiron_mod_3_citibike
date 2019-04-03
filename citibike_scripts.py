import pandas as pd
from mapbox import Geocoder
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import GridSearchCV
import itertools

def set_geocoder(token):
    '''
    takes in your secret_token and returns your usable geocoder
    '''
    return Geocoder(access_token=token)

def get_neighborhood(row):
    '''
    uses the 'end station longitude' and 'end station latitude' of each row to find 
    the neighborhood from the mapbox. Mapbox returns a json file, which we filter 
    through in the second line.
    '''
    response = geocoder.reverse(lon=row['end station longitude'], lat=row['end station latitude'])
    return response.geojson()['features'][0]['context'][0]['text']


def change_birth_year(row,med_age_df):
    '''
    This function will be applied to each row, where ever there is a 1969 we will replace it 
    with the birth year from median_birthyear_no_69 which matches with our stop neighborhood
    '''
    if row['birth year'] == 1969:
        year = med_age_df[med_age_df['stop_nhbr']== row['stop_nhbr']]['birth year']
        return year.item()
    else:
        return row['birth year']
    
def add_neighborhoods(df,end_or_start):
    
    '''
    takes in the data frame and whether its for stop or start stations, then it will create a 
    DataFrame with all of the uniqe stops and thier neighborhood found using the get_neighborhood()
    function defined above.
    '''
    
    #create a dataframe of all unique stations
    station_names = df[end_or_start+' station name'].value_counts() 
    
    #create a list of the keys
    stn_names_li = list(station_names.keys()) 
        
    stops_df = pd.DataFrame(stn_names_li)
    stops_df.rename(columns={0:end_or_start+'_station_name'},inplace=True)

    if end_or_start == 'end':
        
        #concatinate the stop names with the latitude and longitude and drop unneeded columns
        stops_w_latlong = pd.concat([stops_df, df], axis=1, join='inner')
        stops_w_latlong = stops_w_latlong[['end station latitude','end station longitude',
                                           'start station name']] 
        
        stops_w_latlong['end_nhbr'] = stops_w_latlong.apply(lambda row:
                                                            get_neighborhood(row,'end'), axis=1)
       
        return stops_w_latlong
    
    elif end_or_start == 'start':

        stops_w_latlong = pd.concat([stops_df, df], axis=1, join='inner')
        stops_w_latlong = stops_w_latlong[['start station latitude','start station longitude',
                                           'start station name']]
        stops_w_latlong['start_nhbr'] = stops_w_latlong.apply(lambda row: 
                                                              get_neighborhood(row,'start'), axis=1)
        
        return stops_w_latlong


def show_cf(y_true, y_pred, class_names=None, model_name=None):
    plt.figure(figsize=(20,12))
    cf = confusion_matrix(y_true, y_pred)
    plt.imshow(cf, cmap=plt.cm.Blues)
    
    if model_name:
        plt.title("Confusion Matrix: {}".format(model_name))
    else:
        plt.title("Confusion Matrix")
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    class_names = set(y_true)
    tick_marks = np.arange(len(class_names))
    if class_names:
        plt.xticks(tick_marks, class_names)
        plt.yticks(tick_marks, class_names)
    
    thresh = cf.max() / 2.
    
    for i, j in itertools.product(range(cf.shape[0]), range(cf.shape[1])):
        plt.text(j, i, cf[i, j], horizontalalignment='center', color='white' if cf[i, j] > thresh else 'black')
    plt.xticks(rotation=90)
    plt.colorbar()
    
def plot_feature_importances(model, X_train):
    n_features = X_train.shape[1]
    plt.figure(figsize=(8,8))
    plt.barh(range(n_features), model.feature_importances_, align='center') 
    plt.yticks(np.arange(n_features), X_train.columns.values) 
    plt.xlabel("Feature importance")
    plt.ylabel("Feature")
    
def svc_param_selection(X, y, nfolds, kern=str):
    Cs = [0.001, 0.01, 0.1, 1, 10]
    gammas = [0.001, 0.01, 0.1, 1]
    param_grid = {'C': Cs, 'gamma' : gammas}
    grid_search = GridSearchCV(svm.SVC(kernel=kern), param_grid, cv=nfolds)
    grid_search.fit(X, y)
    grid_search.best_params_
    return grid_search.best_params_

def svm_function(X, y, Xt, yt, kern, gamma1, rand, c1=1):
    clf = svm.SVC(kernel=kern, random_state=rand, C=c1)
    clf.fit(X, y)
    training_preds = clf.predict(X)
    val_preds = clf.predict(Xt)
    training_accuracy = accuracy_score(y, training_preds)
    val_accuracy = accuracy_score(yt, val_preds)
    print("Training Accuracy: {:.4}%".format(training_accuracy * 100))
    print("Validation accuracy: {:.4}%".format(val_accuracy * 100))
    return show_cf(yt, val_preds)

def ran_function(X, y, Xt, yt, n_est, rand, crit='gini', maxd=5):
    clf = RandomForestClassifier(n_estimators=n_est, random_state=rand, criterion=crit, max_depth=maxd)
    clf.fit(X, y)
    training_preds = clf.predict(X)
    val_preds = clf.predict(Xt)
    training_accuracy = accuracy_score(y, training_preds)
    val_accuracy = accuracy_score(yt, val_preds)
    print("Training Accuracy: {:.4}%".format(training_accuracy * 100))
    print("Validation accuracy: {:.4}%".format(val_accuracy * 100))
    return show_cf(yt, val_preds), plot_feature_importances(clf, X)

def fix_emd_neighborhood(row):
    if row['end_nhbr_x'] and (type(row['end_nhbr_x']) != float):
        return row['end_nhbr_x']
    elif  row['end_nhbr_y'] and (type(row['end_nhbr_y']) != float):
        return row['end_nhbr_y']
    else:
        return 'What'