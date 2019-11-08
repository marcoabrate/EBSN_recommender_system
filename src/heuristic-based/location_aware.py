import csv
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap

from sklearn.neighbors import KernelDensity

from ..data_utils import DataManagerLocation

DEBUG = True

def visualize_on_map(train_events_latlong, kde):
    """ Visualize the KDE on a map
        Args:
            train_events_latlong: numpy array (n_samples, 2)
            kde: KDE model
    """
    minlat, maxlat = np.min(train_events_latlong[:,0]), np.max(train_events_latlong[:,0])
    minlong, maxlong = np.min(train_events_latlong[:,1]), np.max(train_events_latlong[:,1])
    minlat -= np.abs(minlat)*0.1
    maxlat += np.abs(maxlat)*0.1
    minlong -= np.abs(minlong)*0.1
    maxlong += np.abs(maxlong)*0.1
    latgrid = np.linspace(minlat, maxlat, num=np.abs(maxlat - minlat)*15)
    longgrid = np.linspace(minlong, maxlong, num=np.abs(maxlong - minlong)*15)

    # Set up the data grid for the contour plot
    X, Y = np.meshgrid(longgrid, latgrid[::-1])
    #land_reference = data.coverages[6][::5, ::5]
    #land_mask = (land_reference > -9999).ravel()
    xy = np.vstack([Y.ravel(), X.ravel()]).T # this is lat long
    #xy = np.radians(xy[land_mask])

    fig = plt.figure(figsize=(19.2, 10.8))
    m = Basemap(projection='cyl', resolution='l',
                llcrnrlat=minlat, urcrnrlat=maxlat,
                llcrnrlon=minlong, urcrnrlon=maxlong)
    m.drawmapboundary(fill_color='#DDEEFF')
    m.drawcoastlines()
    m.drawcountries()
    
    Z = np.exp(kde.score_samples(np.radians(xy)))
    Z = Z.reshape(X.shape)
    
    levels = np.linspace(0, Z.max(), 50)
    m.contourf(X, Y, Z, levels=levels, cmap="Reds") # drawing gaussian kernels
    m.scatter(train_events_latlong[:,1], train_events_latlong[:,0], zorder=3,
              cmap="rainbow", latlon=True)

def geo_profile_recommender(dict_user_event_train, dict_event_latlong,
    test_events, test_user):#, bandwidth):
    """ Recommender based on the location of past events attended by test user
        Args:
            dict_user_event_train: user id ->
                {"event_id_list": list of training event ids, "rsvp_time_list": list of RSVPs}
            dict_user_latlong: user id -> (latitude, longitude)
            dict_event_latlong: event id -> (latitude, longitude)
            bandwidth: the bandwidth for the KDE algorithm
        Return:
            dict_event_score: event id -> score
    """
    # creating numpy array of test events latitude and longitude
    # to score them later, with kde.score_samples
    test_events_latlong = np.zeros((len(test_events), 2))
    dict_idx_eventid = {}
    for idx, e in enumerate(test_events):
        dict_idx_eventid[idx] = e
        test_events_latlong[idx, 0] = dict_event_latlong[e][0] # latitude
        test_events_latlong[idx, 1] = dict_event_latlong[e][1] # longitude

    # getting the events the test user attended
    train_events = dict_user_event_train[test_user]['event_id_list']
    if len(train_events) < 1 : return {}
   
    # creating numpy array of training events latitude and longitude
    # to give them to kde.fit in orther to create the model
    train_events_latlong = np.zeros(( len(train_events), 2 ))
    for idx, e in enumerate(train_events):
        train_events_latlong[idx, 0] = dict_event_latlong[e][0] # latitude
        train_events_latlong[idx, 1] = dict_event_latlong[e][1] # longitude

    '''
    from sklearn.model_selection import GridSearchCV
    from sklearn.model_selection import LeaveOneOut
    bandwidths = 10 ** linspace(-2, -1, 100)
    grid = GridSearchCV(KernelDensity(metric='haversine', kernel='gaussian'),
            {'bandwidth': bandwidths},
            cv=LeaveOneOut())
    grid.fit(train_es_latlong)
    bandwidth = grid.best_params_['bandwidth']
    '''
    bandwidth = 0.3
    kde = KernelDensity(bandwidth=bandwidth, metric='haversine', kernel='gaussian')
    kde.fit(np.radians(train_events_latlong)) # (n_samples, n_features)
    
    test_events_score = np.exp(kde.score_samples(np.radians(test_events_latlong)))
    
    if DEBUG:
        visualize_on_map(train_events_latlong, kde)
    
    # normalization to 0-1
    test_events_score = np.divide(test_events_score - np.min(test_events_score),
        np.max(test_events_score) - np.min(test_events_score))
    
    dict_event_score = {}
    for idx, e in dict_idx_eventid.items():
        dict_event_score[e] = test_events_score[idx]

    return dict_event_score

if __name__ == "__main__":

    city = "lausanne"
    partition_num = 0
    dm = DataManagerLocation(city, partition_num)
    
    # Dictionary:
    #     user_id -> {"event_id_list": list of training event ids, "rsvp_time_list": list of RSVPs}
    dict_user_event_train = dm.get_dict_user_events_train()
    # Dictionary: user id -> (latitude, longitude)
    dict_user_latlong = dm.get_users_latlong()
    # Dictionary: event_id -> (latitude, longitude)
    dict_event_latlong = dm.get_events_latlong()

    # Set of test users
    test_users = dm.get_set_test_users()
    # List of test events
    test_events = dm.get_list_test_events()

    for test_user in test_users:
        dict_event_score = geo_profile_recommender(dict_user_event_train,
                                                   dict_event_latlong,
                                                   test_events,
                                                   test_user)
    return
