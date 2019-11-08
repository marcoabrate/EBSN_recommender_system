import logging
import csv
from os import path

from model import EventContentModel
from user_profiles import UserProfileSum, UserProfileTimeWeighted, UserProfileInversePopularity

from ..data_utils import DataManager

# Define the Logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(name)s : %(message)s',
                    level=logging.INFO)
LOGGER = logging.getLogger('content_based.event_recommender')
LOGGER.setLevel(logging.INFO)

MAX_RECS_PER_USER = 100

class ContentBasedModelConf(object):

    def __init__(self, algorithm, hyper_parameters, pre_processment, input_data):
        """ Model for the Content Based event recommender
            Args:
                algorithm: "TFIDF", "LSI" (faster), "LDA" (more accurate)
                hyper_parameters: Dictionary, keys: num_topics, num_corpus_passes, num_iterations
                pre_processment: "text" -> ["replace_numbers_with_spaces"]
                    "word" -> ["get_stemmed_words", "strip_punctuations", "remove_stop_words"]
                input_data: column names for creating the corpus ("name", "description", ...)
        """
        self.algorithm = algorithm
        self.hyper_parameters = hyper_parameters
        self.pre_processment = pre_processment
        self.input_data = input_data

class PostProcessConf(object):

    def __init__(self, types, params):
        """ Configuration class for the post processment of the corpus
            Args:
                types: can be ["filter_extreme_words"]
                params: "no_below_freq" -> int
        """
        self.types = types
        self.params = params

class UserProfileConf(object):

    def __init__(self, name, params):
        """ Configuration class for the user profile
            Args:
                name: "SUM", "TIME", "INV-POPULARITY"
                params: "daily_decay" -> float
        """
        self.name = name
        self.params = params

def cb_train(cb_model_conf, post_process_conf, dm):
    """ Train the Content Based model
        Args:
            cb_model_conf: ContentBasedModelConf class
            post_process_conf: PostProcessConf class
            dm: DataManagerContent class
        Return:
            event_cb_model: EventContentModel class
            dict_event_content: event id -> selected columns content (e.g.
                Dictionary: name -> name of the event, description -> description of the event)
    """
    LOGGER.info("Creating Model [%s]", cb_model_conf.algorithm)
    event_cb_model = EventContentModel(cb_model_conf.pre_processment,
                                       cb_model_conf.algorithm,
                                       cb_model_conf.hyper_parameters)

    # read the Corpus
    filename = dm.get_events_name_desc_filename() 
    LOGGER.info("Reading Corpus from [%s]", filename)
    dict_event_content = event_cb_model.read_and_pre_process_corpus(filename, cb_model_conf.input_data)

    # post process Corpus
    event_cb_model.post_process_corpus(post_process_types=post_process_conf.types,
                                       params=post_process_conf.params,
                                       dict_event_content=dict_event_content,
                                       content_columns=cb_model_conf.input_data)

    # train the model
    LOGGER.info("Training the Model")
    event_cb_model.train_model()

    return event_cb_model, dict_event_content

def cb_profile_recommender(event_cb_model, user_profile_conf, dict_user_event_train, dict_event_content,
    test_user, dict_count_rsvps_events_train, partition_time):
    """ Recommend events to users based on the content of past events
        Args:
            event_cb_model: EventContentModel class
            user_profile_conf: UserProfileConf class
            dict_user_event_train: user id ->
                {"event_id_list": list of training event ids, "rsvp_time_list": list of RSVPs}
            dict_event_content: event id -> selected columns content (e.g.
                Dictionary: name -> name of the event, description -> description of the event)
            test_user: test user id
            dict_count_rsvps_events_train: Dictionary: event id -> number of RSVPs
            partition_time: Integer: time when the partition is
        Return:
            Dictionary: event id -> score
    """
    if user_profile_conf.name == 'SUM':
        # Create the User Profile
        user_profile = UserProfileSum(user_profile_conf.params,
                                      dict_user_events_train[test_user],
                                      event_cb_model, dict_event_content)
        # Submit the query passing the User Profile Representation
        return event_cb_model.query_model(user_profile.get())

    elif user_profile_conf.name == 'TIME':
        # Add the partition time to the user_event_train data
        dict_user_events_train[user]['partition_time'] = partition_time
        # Create the User Profile
        user_profile = UserProfileTimeWeighted(user_profile_conf.params,
                                               dict_user_events_train[test_user],
                                               event_cb_model,
                                               dict_event_content)
        # Submit the query passing the User Profile Representation
        return event_cb_model.query_model(user_profile.get())

    elif user_profile_conf.name == 'INV-POPULARITY':
        # Add the rsvp_count_list to the train events
        dict_user_events_train[user]['rsvp_count_list'] =\
            [dict_count_rsvps_events_train.get(event_id, 0)
             for event_id in dict_user_events_train[user]['event_id_list']]
        # Create the User Profile
        user_profile = UserProfileInversePopularity(user_profile_conf.params,
                                                    dict_user_events_train[test_user],
                                                    event_cb_model,
                                                    dict_event_content)
        # Submit the query passing the User Profile Representation
        return event_cb_model.query_model(user_profile.get())

if __name__ == "__main__":

    cb_model_conf = ContentBasedModelConf(algorithm='TFIDF',
                                          hyper_parameters={'num_topics': 3,
                                                            'num_corpus_passes' : 1,
                                                            'num_iterations': 20},
                                          pre_processment={"text": ["replace_numbers_with_spaces"],
                                                           "word": ["strip_punctuations",
                                                                    "remove_stop_words"]},
                                          input_data=["name", "description"])
    
    post_process_conf = PostProcessConf(types=[],
                                        params={})

    user_profile_conf = UserProfileConf(name="SUM",
                                        params={"daily_decay": 0.01},
                                        params_name="name")

    city = "lausanne"
    partition_num = 0
    dm = DataManagerContent(city, partition_num)
    result_dir = "results/"# directory where to save results
    
    # train
    event_cb_model, dict_event_content = cb_train(cb_model_conf, post_process_conf, dm)
    print(dict_event_content)
    print(event_cb_model)
    
    LOGGER.info("Reading the Partition data (test users, test events and train user-event pairs)")
    # Set of test users
    test_users = dm.get_set_test_users()
    # List of test events
    test_events = dm.get_list_test_events()
    # Dictionary: user id ->
    #     {"event_id_list": list of training event ids, "rsvp_time_list": list of RSVPs}
    dict_user_events_train = dm.get_dict_user_events_train()
    # Dictionary: event id -> number of RSVPs
    dict_count_rsvps_events_train = dm.get_dict_event_rsvp_count_train(dict_user_events_train)
    # Integer: time when the partition is
    partition_time = dm.get_partition_time()
    
    LOGGER.info("Creating the Index to submit the User Profile Queries")
    # event_cb_model : EventContentModel
    event_cb_model.index_events(test_events)

    for user_count, test_user in enumerate(test_users):
        # Log progress
        if user_count % 1000 == 0:
            LOGGER.info("PROGRESS: at user #%d", user_count)

        # The user receives no recommendation if it doesn't have at least one event in train
        if test_user not in dict_user_events_train:
            continue

        dict_event_score = cb_profile_recommender(event_cb_model,
                                                  user_profile_conf,
                                                  dict_user_event_train,
                                                  dict_event_content,
                                                  test_user,
                                                  dict_count_rsvps_events_train,
                                                  partition_time)

    return
