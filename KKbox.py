import numpy as np
import pandas as pd #used to import csv files easily
import lightgbm as lgb #gradient boosted tree builder
from sklearn.model_selection import train_test_split #used to easily split data into feature vectors and labels

data_folder = './data/'

# First we want to load the data from csv files into pandas dataframes for easy handling
# We want pandas dataframes because we can manipulate them more easily than other types of objects
def load_data():
	train = pd.read_csv(data_folder + 'train.csv', dtype={'msno' : 'category',
                                                'source_system_tab' : 'category',
                                                  'source_screen_name' : 'category',
                                                  'source_type' : 'category',
                                                  'target' : np.uint8,
                                                  'song_id' : 'category'})
	test = pd.read_csv(data_folder + 'test.csv', dtype={'msno' : 'category',
                                                'source_system_tab' : 'category',
                                                'source_screen_name' : 'category',
                                                'source_type' : 'category',
                                                'song_id' : 'category'})
	members = pd.read_csv(data_folder + 'members.csv',dtype={'city' : 'category',
                                                      'bd' : np.uint8,
                                                      'gender' : 'category',
                                                      'registered_via' : 'category'},
                     parse_dates=['registration_init_time','expiration_date'])
	songs = pd.read_csv(data_folder + 'songs.csv',dtype={'genre_ids': 'category',
                                                  'language' : 'category',
                                                  'artist_name' : 'category',
                                                  'composer' : 'category',
                                                  'lyricist' : 'category',
                                                  'song_id' : 'category'})
	songs_extra = pd.read_csv(data_folder + 'song_extra_info.csv')
	return train, test, members, songs, songs_extra


# All the information in songs can be concatenated to train and test based on song_id
# Therefore, we use the pandas merge function to combine songs and train/test by song_id
def merge_songs(train, test, songs):
	train = train.merge(songs, how='left', on='songs_id')
	test = test.merge(songs, how='left', on='songs_id')
	return train, test

# members has the registration time and expiration year fields in single integer format with no separations 
# Ex: November 25, 2017 is listed as 20171125, so we use datetime.year/month/day to separate these
# We also calculate the number of days the user has been a member. 
def fix_members(members):
	members['registration_year'] = members['registration_init_time'].dt.year
	members['registration_month'] = members['registration_init_time'].dt.month
	members['registration_date'] = members['registration_init_time'].dt.day

	members['expiration_year'] = members['expiration_date'].dt.year
	members['expiration_month'] = members['expiration_date'].dt.month
	members['expiration_date'] = members['expiration_date'].dt.day
	members = members.drop(['registration_init_time'], axis=1) #we dont need these fields anymore
	members = members.drop(['expiration_date'], axis=1)
	return members

if __name__ == "__main__":
	train, test, members, songs, songs_extra = load_data()
	train, test = merge_songs(train, test, songs)
	members = fix_members(members)