import logging
import csv

from os import path

from model import EventContentModel

from user_profiles import UserProfileSum, UserProfileTimeWeighted, UserProfileInversePopularity 

# Define the Logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(name)s : %(message)s',
                    level=logging.INFO)
LOGGER = logging.getLogger('content_based.event_recommender')
LOGGER.setLevel(logging.INFO)

MAX_RECS_PER_USER = 100

##############################################################################
# PRIVATE FUNCTIONS
##############################################################################
def _get_set_test_users(partition_dir):
    """ Read the Test Users CSV file """
    test_users_csv_path = path.join(partition_dir, "users_test.csv")
    with open(test_users_csv_path, "r") as users_test_file:
        users_test_reader = csv.reader(users_test_file, delimiter=",")
        return set([row[0] for row in users_test_reader])

def _get_list_test_events(partition_dir):
    """ Read the Test Events CSV file """
    test_events_csv_path = path.join(partition_dir, "event-candidates_test.csv")
    with open(test_events_csv_path, "r") as event_test_file:
        events_test_reader = csv.reader(event_test_file, delimiter=",")
        return [row[0] for row in events_test_reader]

def _get_dict_user_events_train(partition_dir):
    """ Read the Train User-Event CSV file """
    train_user_events_csv_path = path.join(partition_dir, "user-event-rsvptime_train.csv")
    with open(train_user_events_csv_path, "r") as user_event_train_file:
        user_event_train_reader = csv.reader(user_event_train_file, delimiter=",")
        next(user_event_train_reader)
        dict_user_event = {}
        for row in user_event_train_reader:
            user_id = row[0]
            event_id = row[1]
            rsvp_time = int(row[2]) # what is this rsvp???
            # maybe if marco participated to event 2 (RSVP yes) 5 days ago, RSVP is five
            dict_user_event.setdefault(user_id, {'event_id_list': [], 'rsvp_time_list': []})
            dict_user_event[user_id]['event_id_list'].append(event_id)
            dict_user_event[user_id]['rsvp_time_list'].append(rsvp_time)
        return dict_user_event

def _get_dict_event_rsvp_count_train(partition_dir):
    """ Read the count_users_per_train-event_train.csv  """
    dict_event_count = {}
    count_rsvps_filename = path.join(partition_dir, "..", "count_users_per_test-event_train.csv")
    with open(count_rsvps_filename, "r") as count_rsvps_file:
        count_rsvps_reader = csv.reader(count_rsvps_file, delimiter=",")
        next(count_rsvps_reader)
        for row in count_rsvps_reader:
            dict_event_count.setdefault(row[0], int(row[1]))
    return dict_event_count

def _get_partition_time(partition_dir, partition_number):
    """ Read the Partition Times CSV File and extract the partition_time """
    partition_times_path = path.join(partition_dir, "..", "..", "..", "partition_times.csv")
    # this csv file is probably as follow:
    # -------------------------------------
    # | partition_number | partition_time |
    # -------------------------------------
    partition_time = None
    with open(partition_times_path, "r") as partition_times_file:
        partition_times_reader = csv.reader(partition_times_file)
        partition_times_reader.next()
        # _ here we can understand that there is one partition time for each partition
        for row in partition_times_reader:
            if int(row[0]) == partition_number:
                partition_time = int(row[2])

    return partition_time

class ContentBasedModelConf(object):

    def __init__(self, algorithm, hyper_parameters, params_name, pre_processment, input_data):
        self.algorithm = algorithm
        self.hyper_parameters = hyper_parameters
        self.params_name = params_name
        self.pre_processment = pre_processment
        self.input_data = input_data

class PostProcessConf(object):

    def __init__(self, name, params_name, types, params):
        self.name = name
        self.params_name = params_name
        self.types = types
        self.params = params

class UserProfileConf(object):

    def __init__(self, name, params, params_name):
        self.name = name
        self.params = params
        self.params_name = params_name

def cb_train(cb_model_conf, post_process_conf, partition_dir):
    """
    Function that trains and recommend events to all test users
    For that it uses an EventContentModel object
    """

    LOGGER.info("Creating Model [%s]", cb_model_conf.algorithm)
    event_cb_model = EventContentModel(cb_model_conf.pre_processment,
                                       cb_model_conf.algorithm,
                                       cb_model_conf.hyper_parameters,
                                       cb_model_conf.params_name)

    # Read the Corpus
    filename = path.join(partition_dir, "event-name-desc_all.csv")
    LOGGER.info("Reading Corpus from [%s]", filename)
    dict_event_content = event_cb_model.read_and_pre_process_corpus(filename, cb_model_conf.input_data)

    # Post Process Corpus
    event_cb_model.post_process_corpus(post_process_types=post_process_conf.types,
                                       params=post_process_conf.params,
                                       dict_event_content=dict_event_content,
                                       content_columns=cb_model_conf.input_data)

    # Train the model
    LOGGER.info("Training the Model")
    event_cb_model.train_model()

    return event_cb_model, dict_event_content

def cb_recommend(event_cb_model, user_profile_conf, dict_event_content,
                 partition_dir, partition_number):
    """
    Recommend events to users
    Given a trained Content Based Model are generated N recommendations to the same User
    One recommendation for each user profile type (i.e. for each personalization approach)
    """
    LOGGER.info("Reading the Partition data (test users, test events and train user-event pairs)")
    test_users = _get_set_test_users(partition_dir) # list of user IDs
    test_events = _get_list_test_events(partition_dir) # list of events IDs
    dict_user_events_train = _get_dict_user_events_train(partition_dir)
    # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    # | user_id | list of event_ids | list of RSVP times|

    LOGGER.info("Reading extra data for User Profiles")
    dict_count_rsvps_events_train = _get_dict_event_rsvp_count_train(partition_dir)
    # ^^^^^^^^^^^^^^^^^^^
    # | event_id | RSVPs|

    #partition_time = _get_partition_time(partition_dir, partition_number) # what is this partition time?
    # ^it is just an integer

    LOGGER.info("Creating the Index to submit the User Profile Queries")
    # event_cb_model : EventContentModel
    event_cb_model.index_events(test_events)

    LOGGER.info("Recommending the Test Events to Test Users")
    dict_user_rec_events = {}
    user_count = 0
    for user in test_users:
        # Log progress
        if user_count % 1000 == 0:
            LOGGER.info("PROGRESS: at user #%d", user_count)
        user_count += 1

        # Every user has at least an empty recommendation
        dict_user_rec_events.setdefault(user, [])

        # The user receives no recommendation if it doesn't have at least one event in train
        if user not in dict_user_events_train:
            continue

        # -------------------------------------------------------------------------
        # Call the Recommendation function based on the User Profile Type

        if user_profile_conf.name == 'SUM':
            # Create the User Profile
            user_profile = UserProfileSum(user_profile_conf.params, dict_user_events_train[user],
                                          event_cb_model, dict_event_content)
            # Submit the query passing the User Profile Representation
            dict_user_rec_events[user] = event_cb_model.query_model(
                                            user_profile.get(), test_events,
                                            dict_user_events_train[user]['event_id_list'],
                                            MAX_RECS_PER_USER)

        elif user_profile_conf.name == 'TIME':
            # Add the partition time to the user_event_train data
            dict_user_events_train[user]['partition_time'] = partition_time
            # Create the User Profile
            user_profile = UserProfileTimeWeighted(
                                user_profile_conf.params, dict_user_events_train[user],
                                event_cb_model, dict_event_content)
            # Submit the query passing the User Profile Representation
            dict_user_rec_events[user] = event_cb_model.query_model(
                                            user_profile.get(), test_events,
                                            dict_user_events_train[user]['event_id_list'],
                                            MAX_RECS_PER_USER)

        elif user_profile_conf.name == 'INV-POPULARITY':
            # Add the rsvp_count_list to the train events
            dict_user_events_train[user]['rsvp_count_list'] =\
                [dict_count_rsvps_events_train.get(event_id, 0)
                    for event_id in dict_user_events_train[user]['event_id_list']]
            # Create the User Profile
            user_profile = UserProfileInversePopularity(
                            user_profile_conf.params, dict_user_events_train[user],
                            event_cb_model, dict_event_content)
            # Submit the query passing the User Profile Representation
            dict_user_rec_events[user] = event_cb_model.query_model(
                                            user_profile.get(), test_events,
                                            dict_user_events_train[user]['event_id_list'],
                                            MAX_RECS_PER_USER)

    return dict_user_rec_events

if __name__ == "__main__":

    cb_model_conf = ContentBasedModelConf(algorithm='TFIDF',
                            # LSI much faster, LDA more accurate
                                          hyper_parameters={'num_topics': 3,
                                                            'num_corpus_passes' : 1,
                                                            'num_iterations': 20},
                                          params_name="name",
                                          pre_processment={"text": ["replace_numbers_with_spaces"],
                            # "text": ["replace_numbers_with_spaces"]
                            # "word": ["get_stemmed_words", "strip_punctuations", "remove_stop_words"]},
                                                           "word": ["strip_punctuations",
                                                                    "remove_stop_words"]},
                                          input_data=["name", "description"])
    # ^input_data: when creating a bag-of-word, do we take into consideration name and description?
    # in this case, we consider only the name of the event
    post_process_conf = PostProcessConf(name='NO',
                                        params_name="",
        # types = ["filter_extreme_words"]
                                        types=[],
        # params = {"no_below_freq": int}
                                        params={})
    user_profile_conf = UserProfileConf(name="SUM",
        # name = ["SUM", "TIME", "INV-POPULARITY"] 
                                        params={"daily_decay": 0.01},
                                        params_name="name")

    partition = 1
    region = "phoenix"
    # directory with the data corresponding to "region" and partition number "partition"
    partition_dir = "data/"
    result_dir = "results/"# directory where to save results

    model_profile_name = "%s-%s:%s-%s:%s-%s_%s:%s:%s" % ("CB",
                                                         cb_model_conf.algorithm,
                                                         "PP",
                                                         post_process_conf.name,
                                                         "UP",
                                                         user_profile_conf.name,
                                                         cb_model_conf.params_name,
                                                         post_process_conf.params_name,
                                                         user_profile_conf.params_name)

    # Train
    event_cb_model, dict_event_content = cb_train(cb_model_conf, post_process_conf, partition_dir)
    print(dict_event_content)
    print(event_cb_model)
    
    # Get User Recommendations
    dict_user_rec_events = cb_recommend(event_cb_model, user_profile_conf,
                                        dict_event_content, partition_dir, partition)
    for k in dict_user_rec_events.keys():
        print(k.upper())
        for r in dict_user_rec_events[k]:
            print(r)
        print()
