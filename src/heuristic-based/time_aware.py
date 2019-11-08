import csv
import numpy as np
import matplotlib.pyplot as plt
from dateutil import tz as timezone
from datetime import datetime

from scipy.spatial.distance import cosine

from ..data_utils import DataManagerTime

DEBUG = True

def visualize_user_matrix(dayhour_matrix):
    """ Visualization of the 7x24 day-hour matrix
        Args:
            dayhour_matrix: 7x24 matrix
    """
    return

def _update_dayhour_matrix(dayhour_matrix, start, end, utc_offset):
    """ Return the updated 7x24 day-hour matrix
        Args:
            dayhour_matrix: 7x24 matrix
            start, end, utc_offset: start, end times and UTC offset
    """
    start, end = datetime.fromtimestamp(start/1000, tz=timezone.tzoffset(None, utc_offset/1000)),\
            datetime.fromtimestamp(end/1000, tz=timezone.tzoffset(None, utc_offset/1000))
    # Monday is 0, Sunday is 6
    startday, endday = start.weekday(), end.weekday()
    starthour = start.hour
    endhour = end.hour if end.minute == 0 else end.hour+1
    if startday == endday:
        dayhour_matrix[startday, starthour:endhour] += 1
    else:
        dayhour_matrix[startday, starthour:] += 1
        dayhour_matrix[endday, :endhour] += 1
    return dayhour_matrix

def _create_features(dayhour_matrix):
    """ Return the feature vector, given the 7x24 day-hour matrix
        Args:
            dayhour_matrix: 7x24 matrix
        Return:
            (37,) feature vector
            | daily 7 | weekday_orno 2 | hourly 24 | turns 4 |
    """
    daily = np.sum(dayhour_matrix, axis=1) /24
    weekday_orno = np.array([np.sum(dayhour_matrix[:5])/(5*24), np.sum(dayhour_matrix[5:])/(2*24)])
    hourly = np.sum(dayhour_matrix, axis=0) /7
    turns = np.array([np.sum(dayhour_matrix[:,5:11]),
                               np.sum(dayhour_matrix[:,11:17]),
                               np.sum(dayhour_matrix[:,17:23]),
                               np.sum(dayhour_matrix[:,22:])+np.sum(dayhour_matrix[:,0:5])]) /(7*6)

    return np.concatenate(daily, weekday_orno, hourly, turns)

def _create_user_centroid(test_u, dict_user_event_train, dict_event_startend):
    """ Return the centroid of the selected test user
        Args:
            test_u: test user id
            dict_user_event_train: user id ->
                {"event_id_list": list of training event ids, "rsvp_time_list": list of RSVPs}
            dict_event_startend: event id -> (start, end)
    """
    user_matrix = np.zeros((7, 24))
    list_events = dict_user_event_train[test_u]["event_id_list"]
    for e in list_events:
        start, end, utc_offset = dict_event_startend[e]
        user_matrix = _update_dayhour_matrix(user_matrix, start, end, utc_offset)

    if DEBUG:
        visualize_user_matrix(user_matrix)

    user_matrix /= len(list_events)

    return _create_features(user_matrix)

def time_profile_recommender(dict_user_event_train, dict_event_startend, test_events, test_user):
    """ Recommender based on the time at which past events attended by the test user occured
        Args:
            dict_user_event_train: user id ->
                {"event_id_list": list of training event ids, "rsvp_time_list": list of RSVPs}
            dict_event_startend: event id -> (start, end)
            test_events: list of test events
            test_user: test user id
        Return:
            Dictionary: event id -> score
    """
    user_centroid = _create_user_centroid(test_user, dict_user_event_train, dict_event_startend)

    dict_event_score = {}
    # if computing the similarity for every event takes too
    # much time, we can generate the following matrix and use k-NN:
    # test_events_matrix = np.zeros((len(test_events), len(user_centroid)))
    for e in test_events:
        event_matrix = np.zeros((7, 24))
        start, end, utc_offset = dict_event_startend[e]
        event_matrix = _update_dayhour_matrix(event_matrix, start, end, utc_offset)
        event_feature_vect = _create_features(event_matrix)
        cosine_distance = cosine(user_centroid, event_feature_vect)
        dict_event_score[e] = 1 - cosine_distance # similatiry = 1 - distance

    return dict_event_score

if __name__ == "__main__":

    city = "lausanne"
    partition_num = 0
    dm = DataManagerTime(city, partition_num)

    # Dictionary:
    #     user_id -> {"event_id_list": list of training event ids, "rsvp_time_list": list of RSVPs}
    dict_user_event_train = dm.get_dict_user_events_train()
    # Dictionary: event id -> (start, end)
    dict_event_startend = dm.get_events_startend()

    # Set of test users
    test_users = dm.get_set_test_users()
    # List of test events
    test_events = dm.get_list_test_events()

    for test_user in test_users:
        # The user receives no recommendation if it doesn't have at least one event in train
        if test_user not in dict_user_events_train:
            continue

        dict_event_score = geo_profile_recommender(dict_user_event_train,
                                                   dict_event_latlong,
                                                   test_events,
                                                   test_user)
    return

