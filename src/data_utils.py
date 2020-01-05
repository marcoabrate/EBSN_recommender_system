import unicodecsv as csv
import numpy as np

class DataManager:
    self.data_dir = "../data/"
    
    def __init__(self, city, partition_num):
        self.city = city
        self.partition = partition_num

    def get_dict_user_events_train(self):
        """ Read the Train User-Event CSV file
            ----------------------------------
            | User ID | Event ID | RSVP time |
            ----------------------------------
            Return:
                Dictionary: user id ->
                    {"event_id_list": list of training event ids, "rsvp_time_list": list of RSVPs}
        """
        train_user_events_csv_path = path.join(
            self.data_dir, self.city, "partition"+self.partition_num, "user-event-rsvptime_train.csv")
        with open(train_user_events_csv_path, "r") as user_event_train_file:
            user_event_train_reader = csv.reader(user_event_train_file, delimiter=",")
            if csv.Sniffer().has_header:
                next(user_event_train_reader)
            dict_user_event = {}
            for row in user_event_train_reader:
                user_id = row[0]
                event_id = row[1]
                rsvp_time = int(row[2])
                dict_user_event.setdefault(user_id, {"event_id_list": [], "rsvp_time_list": []})
                dict_user_event[user_id]["event_id_list"].append(event_id)
                dict_user_event[user_id]["rsvp_time_list"].append(rsvp_time)
            return dict_user_event

    def get_set_test_users(self):
        """ Read the Test Users CSV file
            -----------
            | User ID |
            -----------
            Return:
                Set of TEST user ids
        """
        test_users_csv_path = path.join(
            self.data_dir, self.city, "partition"+self.partition_num, "users_test.csv")
        with open(test_users_csv_path, "r") as users_test_file:
            users_test_reader = csv.reader(users_test_file, delimiter=",")
            return set([row[0] for row in users_test_reader])

    def get_list_test_events(self):
        """ Read the Test Events CSV file
            -----------
            | Event ID |
            -----------
            Return:
                Set of TEST event ids
        """
        test_events_csv_path = path.join(
            self.data_dir, self.city, "partition"+self.partition_num, "event-candidates_test.csv")
        with open(test_events_csv_path, "r") as event_test_file:
            events_test_reader = csv.reader(event_test_file, delimiter=",")
            return [row[0] for row in events_test_reader]

class DataManagerContent(DataManager):

    def get_events_name_desc_filename(self):
        return path.join(
            self.data_dir, self.city, "partition"+self.partition_num, "event_name-desc.csv")

    def get_dict_event_rsvp_count_train(self, dict_user_event):
        """ Return the number of RSVPs per training event
            Args:
                dict_user_event: dictionary with user id ->
                    {"event_id_list": list of training event ids, "rsvp_time_list": list of RSVPs}
            Return:
                Dictionary: event id -> number of RSVPs
        """
        dict_event_count = {}
        # TODO: count the number of RSVPs per event id
        return dict_event_count

    def get_partition_time(self):
        """ Read the Partition Times CSV File
            -------------------------------------
            | Partition number | Partition time |
            -------------------------------------
            Return:
                Integer: time when the partition is
        """
        partition_times_path = path.join(
            self.data_dir, self.city, "partition_times.csv")
        partition_time = None
        with open(partition_times_path, "r") as partition_times_file:
            partition_times_reader = csv.reader(partition_times_file)
            partition_times_reader.next()
            for row in partition_times_reader:
                if int(row[0]) == self.partition:
                    partition_time = int(row[1])

        return partition_time

class DataManagerLocation(DataManager):

    def get_events_latlong(self):
        """ Read the Events latitude, longitude information
            -----------------------------------
            | Event ID | Latitude | Longitude |
            -----------------------------------
            Return:
                Dictionary: event id -> (latitude, longitude)
        """
        event_latlong_csv_path = path.join(
            self.data_dir, self.city, "partition"+self.partition_num, "event_latitude_longitude.csv")
        with open(event_latlong_csv_path, "r") as event_latlong_file:
            event_latlong_reader = csv.reader(event_latlong_file, delimiter=",")
            next(event_latlong_reader)
            dict_event_latlong = {}
            for row in event_latlong_reader:
                event_id = row[0]
                lat_ = row[1]
                long_ = row[2]
                dict_event_latlong[event_id] = (lat_, long_)
            return dict_event_latlong
    
    def get_users_latlong(self):
        """ Read the Users latitude, longitude information
            ----------------------------------
            | User ID | Latitude | Longitude |
            ----------------------------------
            Return:
                Dictionary: user id -> (latitude, longitude)
        """
        user_latlong_csv_path = path.join(
            self.data_dir, self.city, "partition"+self.partition_num, "user_latitude_lnogitude.csv")
        with open(user_latlong_csv_path, "r") as user_latlong_file:
            user_latlong_reader = csv.reader(user_latlong_file, delimiter=",")
            next(user_latlong_reader)
            dict_user_latlong = {}
            for row in user_latlong_reader:
                user_id = row[0]
                lat_ = row[1]
                long_ = row[2]
                dict_user_latlong[user_id] = (lat_, long_)
            return dict_user_latlong

class DataManagerTime(DataManager):

    def get_events_startend(self):
        """ Read the Events start, end time information
            ---------------------------------------
            | Event ID | Start | End | UTC offset |
            ---------------------------------------
            Return:
                Dictionary: event id -> (start, end, UTC offset)
        """
        event_startend_csv_path = path.join(
            self.data_dir, self.city, "partition"+self.partition_num, "event_start_end_time.csv")
        with open(event_startend_csv_path, "r") as event_startend_file:
            event_startend_reader = csv.reader(event_startend_file, delimiter=",")
            next(event_startend_reader)
            dict_event_startend = {}
            for row in event_startend_reader:
                event_id = row[0]
                start = row[1]
                end = row[2]
                utc_offset = row[3]
                dict_event_startend[event_id] = (start, end, utc_offset)
            return dict_event_startend
