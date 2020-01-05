import os
import sys
from time import sleep

import unicodecsv as csv
import requests
import requests_oauthlib
import json

def oauth2_session(client_id, client_secret, redirect_uri):
    meetup = requests_oauthlib.OAuth2Session(
        client_id,
        redirect_uri = redirect_uri)

    params = {
        #"client_id": "???"
        "response_type": "anonymous_code",
        "redirect_uri": redirect_uri}

    url, state = meetup.authorization_url(
            "https://secure.meetup.com/oauth2/authorize",
            response_type="anonymous_code")
    print(url)
    # TODO: get the code from the url

    token = meetup.fetch_token(
        "https://secure.meetup.com/oauth2/access",
        client_secret = client_secret,
        grant_type = "anonymous_code",
        code = code)
    
    return access_token, refresh_token

def get_nearby_groups(city, headers):
    
    url = "https://api.meetup.com/find/groups?"
    payload = {
        "location": city,
        "radius": smart,
        "order": "most_active"
        }

    dic_catid_cat = {}
    dic_metacatid_metacat = {}

    print(f"\n[*] getting the Meetup groups in {city}")
    r = requests.get(url, headers = headers, params = payload)
    groups_dic = json.load(r.json())

    with open(city+"/groups_info.csv", "wb") as csvgroups:
        groups_writer = csv.writer(csvgroups, delimiter=",")
        groups_writer.write(
            "group_id", # int
            "urlname", # string
            "name", # string
            "description", # string
            "created", # int
            "members", # int
            "category_id", # int
            "meta_category_id") # int

        print(f"[*] saving the groups to {city}/groups_info.csv")
        for k in groups_dic.keys():
            kth_group = groups_dic.get(k)
            infos = [
                kth_group.get("id"),
                kth_group.get("urlname"),
                kth_group.get("name"),
                kth_group.get("plain_text_no_images_description"),
                kth_group.get("created"),
                kth_group.get("members")]
            if "category" in kth_group:
                infos.append(kth_group.get("category").get("id"))
                dic_catid_cat.update(kth_group.get("category").get("id"),
                                     kth_group.get("category").get("name"))
            else:
                infos.append(-999)
            if "meta_category" in kth_group:
                infos.append(kth_group.get("meta_category").get("id"))
                dic_metacatid_metacat.update(kth_group.get("meta_category").get("id"),
                                             kth_group.get("meta_category").get("name"))
            else:
                infos.append(-999)

            groups_writer.write(infos)

    print(f"[+] done saving groups")
    return dic_catid_cat, dic_metacatid_metacat

def dump_groups_events(city, headers, no_earlier_than, no_later_than):
    
    api_url = "https://api.meetup.com/"
    payload = {
        "no_later_than": no_earlier_than,
        "no_earlier_than": no_later_than,
        status: "past"}

    with open(city+"/groups_info.csv", "rb") as csvgroups,\
        open(city+"/events_info.csv", "wb") as csvevents:
        groups_reader = csv.reader(csvgroups, delimiter=",")
        groups_reader.next()
        
        events_writer = csv.writer(csvevents, delimiter=",")
        events_writer.write(
            "event_id", # string
            "group_id", # int
            "group_urlname", # string
            "name", # string
            "description", # string
            "time", # int
            "utc_offset",
            "duration", # int
            "latitude", # float
            "longitude", # float
            "yes_rsvp_count") # int

        for row in groups_reader:
            group_id = row[0]
            urlname = row[1]

            url = api_url + urlname + "/events"
            print(f"\n[*] getting the events for the group {urlname}")
            r = requests.get(url, headers = headers, params = payload)
            events_dic = json.load(r.json())
            
            print(f"[*] saving the events to {city}/events_info.csv")
            for k in events_dic.keys():
                kth_event = events_dic.get(k)
                infos = [
                    kth_event.get("id"),
                    group_id,
                    urlname,
                    kth_event.get("name"),
                    kth_event.get("plain_text_no_images_description"),
                    kth_event.get("time"),
                    kth_event.get("utc_offset"),
                    kth_event.get("duration"),
                    kth_event.get("venue").get("lat"),
                    kth_event.get("venue").get("lon"),
                    kth_event.get("yes_rsvp_count")]

                events_writer.write(infos)
            print(f"[+] done saving events")
            # wait 1 second
            sleep(1)

def dump_events_rsvps(city, headers):
    
    api_url = "https://api.meetup.com/"
    payload = {
        "response": "yes"}

    member_ids = set()

    with open(city+"/events_info.csv", "rb") as csvevents,\
        open(city+"/events_rsvps.csv", "wb") as csvrsvps:
        events_reader = csv.reader(csvevents, delimiter=",")
        events_reader.next()
        
        rsvps_writer = csv.writer(csvrsvps, delimiter=",")
        rsvps_writer.write(
            "event_id", # string
            "group_id", # int
            "group_urlname", # string
            "member_id", # int
            "rsvp_time") # int

        for row in events_reader:
            event_id = row[0]
            group_id = row[1]
            urlname = row[2]
            event_name = row[3]

            url = api_url + urlname + "/events/" + event_id + "/rsvps"
            print(f"\n[*] getting the rsvps for the event {event_name}")
            r = requests.get(url, headers = headers, params = payload)
            rsvps_dic = json.load(r.json())
            
            print(f"[*] saving the rsvps to {city}/rsvps_info.csv")
            for k in rsvps_dic.keys():
                kth_rsvp = rsvps_dic.get(k)
                infos = [
                    event_id,
                    group_id,
                    urlname,
                    kth_rsvp.get("member").get("id"),
                    kth_rsvp.get("updated")]

                member_ids.add(kth_rsvp.get("member").get("id"))

                rsvps_writer.write(infos)
            print(f"[+] done saving rsvps")
            # wait 1 second
            sleep(1)
    
    return member_ids

def dump_members_info(city, headers, member_ids):
    api_url = "https://api.meetup.com/"

    with open(city+"/members_info.csv", "wb") as csvmembers:
        members_writer = csv.writer(csvmembers, delimiter=",")
        members_writer.write(
            "member_id", # int
            "name", # string
            "bio", # string
            "latitude", # float
            "longitude") # float

        for memeber_id in member_ids:
            
            url = api_url + "members/" + member_id
            print(f"[*] getting the info for the member {member_id}")
            r = requests.get(url, headers = headers)
            member_info_dic = json.load(r.json())

            infos = [
                member_info_dic.get("id"),
                member_info_dic.get("name")]
            if "bio" in member_info_dic:
                infos.append(member_info_dic.get("bio"))
            else:
                infos.append(-999)
            infos += [
                member_info_dic.get("lat"),
                member_info_dic.get("lon")]

            groups_writer.write(infos)
        # wait 1 second
        sleep(1)

def save_category_ids(city, dic_catid_cat):
    with open(city+"/categories.csv", "wb") as csvcats:
        cats_writer = csv.writer(csvcats, delimiter=",")
        cats_writer.write(
            "category_id", # int
            "name") # string

        for k in dic_catid_cat:
            cats_writer.write(k, dic_catid_cat.get(k))

def save_metacategory_ids(city, dic_metacatid_metacat):
    with open(city+"/metacategories.csv", "wb") as csvmetacats:
        metacats_writer = csv.writer(csvmetacats, delimiter=",")
        metacats_writer.write(
            "meta_category_id", # int
            "name") # string

        for k in dic_metacatid_metacat:
            metacats_writer.write(k, dic_metacatid_metacat.get(k))

if __name__ == '__main__':

    if len(sys.argv) < 2:
        print("[-] usage: python dump_meetup.py {city}\n")
        sys.exit(1)

    city = sys.argv[1]
    
    print(f"[*] working in {city} area, creating directory\n")
    if not os.path.isdir(city):
        os.mkdir(city)

    client_id = r"???"
    client_secret = r"???"
    redirect_uri = "???"

    access_token, refresh_token = oauth2_session(client_id, client_secret, redirect_uri)

    headers = {
        "Authorization": " Bearer "+access_token,
        "Origin": redirect_uri}
    # start time
    no_earlier_than = "2019-01-01T00:00:00.000"
    # end time
    no_later_than = "2019-09-01T00:00:00.000"


    dic_catid_cat, dic_metacatid_metacat = dump_nearby_groups(city, headers)

    save_category_ids(dic_catid_cat)
    save_metacategory_ids(dic_metacatid_metacat)

    dump_groups_events(city, headers, no_earlier_than, no_later_than)

    member_ids = dump_events_rsvps(city, headers)

    dump_members_info(city, headers, member_ids)
