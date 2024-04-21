client_id = "285310898d54476f9b5f27c10f666371"
client_secret = "3eae50b7f4ef437b91a08150287493fd"

import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import json
import csv

song_names = [
    "Shape of You",
    "Uptown Funk",
    "Old Town Road",
    "Blinding Lights",
    "Closer",
    "Dance Monkey",
    "Someone You Loved",
    "Despacito",
    "Sorry",
    "Roar",
    "Havana",
    "Don't Stop Believin'",
    "Hello",
    "Believer",
    "Shake It Off",
    "Happier",
    "Bad Guy",
    "Rockstar",
    "Thinking Out Loud",
    "Rolling in the Deep",
    "Bohemian Rhapsody",
    "Radioactive",
    "Counting Stars",
    "Watermelon Sugar",
    "Sicko Mode",
    "Hotline Bling",
    "Perfect",
    "The Box",
    "All of Me",
    "Love Yourself",
    "Can't Stop the Feeling!",
    "Humble",
    "Sugar",
    "Roses",
    "Stay",
    "Thunder"]
# Initialize Spotipy client
client_credentials_manager = SpotifyClientCredentials(client_id=client_id, client_secret=client_secret)
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

song_ids = []

# Iterate through the list of song names
for song_name in song_names:
    # Search for the song
    results = sp.search(q=song_name, type='track', limit=1)
    for item in results['tracks']['items']:
        # Get the track ID
        song_ids.append(item['id'])


tracks_info = sp.tracks(song_ids)
tracks_audio_features = sp.audio_features(song_ids)

with open('tracks_info.json', 'w') as outfile:
    json.dump(tracks_info, outfile, indent=4)
with open('tracks_features.json', 'w') as outfile:
    json.dump(tracks_audio_features, outfile, indent=4)
    
# Data Preprocessing


# Read the contents of the tracks_info.json file into a dictionary
with open('tracks_info.json', 'r') as file:
    tracks_info = json.load(file)

# Initialize a list to store track details
track_details = []

# Function to fetch album genres based on album ID
def get_album_genres(album_id):
    album_info = sp.album(album_id)
    if 'genres' in album_info:
        return album_info['genres']
    else:
        return []

# Iterate through each track in the tracks_info dictionary
for track in tracks_info['tracks']:
    # Extract relevant information for each track
    track_detail = {
        "album_id": track['album']['id'],
        "album_name": track['album']['name'],
        "album_release_date": track['album']['release_date'],
        "album_total_tracks": track['album']['total_tracks'],
        "album_type": track['album']['album_type'],
        "artist_id": track['artists'][0]['id'],
        "artist_name": track['artists'][0]['name'],
        "artist_type": track['artists'][0]['type'],
        "available_markets": track['available_markets'],
        "disc_number": track['disc_number'],
        "duration_ms": track['duration_ms'],
        "explicit": track['explicit'],
        "isrc": track['external_ids']['isrc'] if 'external_ids' in track else None,
        "id": track['id'],
        "is_local": track['is_local'],
        "name": track['name'],
        "popularity": track['popularity'],
        "track_number": track['track_number']
    }

    # Fetch and add album genres
    album_id = track_detail['album_id']
    album_genres = get_album_genres(album_id)
    track_detail["album_genres"] = album_genres

    # Append the track detail to the track_details list
    track_details.append(track_detail)

# Define the fieldnames for the CSV file
fieldnames = [
    "album_id", "album_name", "album_release_date", "album_total_tracks",
    "album_type", "artist_id", "artist_name", "artist_type", "available_markets",
    "disc_number", "duration_ms", "explicit", "isrc", "id", "is_local",
    "name", "popularity", "track_number", "album_genres"
]

# Write track details to a CSV file
with open('track_details.csv', 'w', newline='') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for track_detail in track_details:
        writer.writerow(track_detail)

print("Track details have been saved to track_details.csv")



import csv

# Read the contents of the tracks_features.json file into a dictionary
with open('tracks_features.json', 'r') as file:
    tracks_features = json.load(file)

# Initialize a list to store audio features
audio_features = []

# Iterate through each track in the tracks_features dictionary
for track in tracks_features:
    # Extract only the required features for each track
    track_features = {
        "danceability": track['danceability'],
        "energy": track['energy'],
        "key": track['key'],
        "loudness": track['loudness'],
        "mode": track['mode'],
        "speechiness": track['speechiness'],
        "acousticness": track['acousticness'],
        "instrumentalness": track['instrumentalness'],
        "liveness": track['liveness'],
        "valence": track['valence'],
        "tempo": track['tempo'],
        "id": track['id'],
        "duration_ms": track['duration_ms'],
        "time_signature": track['time_signature']
    }
    # Append the track features to the audio_features list
    audio_features.append(track_features)

# Define the fieldnames for the CSV file
fieldnames = [
    "danceability", "energy", "key", "loudness", "mode", "speechiness",
    "acousticness", "instrumentalness", "liveness", "valence", "tempo",
    "id", "duration_ms", "time_signature"
]

# Write audio features to a CSV file
with open('track_audio_features.csv', 'w', newline='') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for track_feature in audio_features:
        writer.writerow(track_feature)

print("Audio features have been saved to track_audio_features.csv")


#Audio features have been saved to track_audio_features.csv


import csv

# Read track details from CSV
track_details = []
with open('track_details.csv', 'r', newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        track_details.append(row)

# Read audio features from CSV
audio_features = []
with open('track_audio_features.csv', 'r', newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        audio_features.append(row)

# Merge data based on track ID
merged_data = []
for detail in track_details:
    for feature in audio_features:
        if detail['id'] == feature['id']:
            merged_data.append({**detail, **feature})
            break

# Define fieldnames for the merged CSV
fieldnames = [
    "album_id", "album_name", "album_release_date", "album_total_tracks",
    "album_type", "artist_id", "artist_name", "artist_type", "available_markets",
    "disc_number", "duration_ms", "explicit", "isrc", "is_local",
    "name", "popularity", "track_number", "album_genres",
    "danceability", "energy", "key", "loudness", "mode", "speechiness",
    "acousticness", "instrumentalness", "liveness", "valence", "tempo",
    "id", "time_signature"
]

# Write merged data to a new CSV file
with open('merged_track_data.csv', 'w', newline='') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for item in merged_data:
        writer.writerow(item)

print("Merged track data has been saved to merged_track_data.csv")


#Merged track data has been saved to merged_track_data.csv

# Feature Engineering

import pandas as pd

# Read the CSV file into a DataFrame
df = pd.read_csv("songs.csv")

# Display missing values
missing_values = df.isnull().sum()
print(missing_values)


from sklearn.preprocessing import MinMaxScaler, LabelEncoder

df["song"] = df["track_name"] + df["artist_name"]

# Remove 'lyrics' and 'len' columns
df.drop(['lyrics', 'len'], axis=1, inplace=True)

# One-hot encoding for 'topic', 'artist_name', and 'genre' columns
df = pd.get_dummies(df, columns=['topic', 'artist_name', 'genre'])

# Min-max normalization for 'release_date' column
scaler = MinMaxScaler()
df['release_date'] = scaler.fit_transform(df[['release_date']])

# Display the modified DataFrame
print(df.head())


# Get all column names from the DataFrame
all_columns = df.columns

# Filter out the column names that start with "topic"
topic_columns = [col for col in all_columns if col.startswith("topic")]

print(topic_columns)


# Get all column names from the DataFrame
all_columns = df.columns

# Filter out the column names that start with "topic"
topic_columns = [col for col in all_columns if col.startswith("genre")]

print(topic_columns)

# Content Based Filtering
#Using content based filterting on specific columns to get 2000 most matching songs

import numpy as np

# Define a custom similarity function
def custom_similarity(song1, song2):
    # Extract features from songs
    features_song1 = np.array([song1['age']] + [song1[col] for col in topic_genre_columns])
    features_song2 = np.array([song2['age']] + [song2[col] for col in topic_genre_columns])

    # Calculate Euclidean distance between the feature vectors
    distance = np.linalg.norm(features_song1 - features_song2)

    return distance

# Get the column names for 'topic' and 'genre' after one-hot encoding
topic_genre_columns = [col for col in df.columns if col.startswith("topic") or col.startswith("genre")]

# Function to find the index of a song by its name
def find_song_index(song_name):
    return df.index[df["song"] == song_name].tolist()

# Input song info
input_song_name 
input_artist_name



input_song_name = input_song_name+input_artist_name

# Find the index of the input song
input_song_index = find_song_index(input_song_name)
if not input_song_index:
    print("Song not found.")
    exit()
print(input_song_index)
input_song_index = input_song_index[0]  # If multiple occurrences found, take the first one

# Get features of the input song
input_song_features = df.iloc[input_song_index]

# Calculate similarity scores between input song and all other songs
similarity_scores = [(i, custom_similarity(input_song_features, df.iloc[i])) for i in range(len(df))]

# Sort songs based on similarity scores
sorted_songs = sorted(similarity_scores, key=lambda x: x[1])

# Select the top 2000 most similar songs
top_2000_songs = sorted_songs[:2000]

# First KNN layer to prioritize songs based upon context

from sklearn.neighbors import NearestNeighbors

# Define the columns for the first KNN
first_knn_columns = ['dating', 'violence', 'world/life', 'night/time', 'shake the audience', 'family/gospel',
                     'romantic', 'communication', 'obscene', 'music', 'movement/places', 'light/visual perceptions',
                     'family/spiritual', 'like/girls', 'sadness', 'feelings']

# Extract the features for the first KNN
X_first_knn = df[first_knn_columns]

# Instantiate the first KNN model
first_knn = NearestNeighbors(n_neighbors=100)
first_knn.fit(X_first_knn)

# Find the indices of nearest neighbors to the input song using the first KNN
input_song_neighbors_first = first_knn.kneighbors([X_first_knn.iloc[input_song_index]], return_distance=False)[0]


# Second KNN layer to prioritize songs based upon the musical factors

# Define the columns for the second KNN
second_knn_columns = ['danceability', 'loudness', 'acousticness', 'instrumentalness', 'valence', 'energy']

# Extract the features for the second KNN
X_second_knn = df.iloc[input_song_neighbors_first][second_knn_columns]

# Instantiate the second KNN model
second_knn = NearestNeighbors(n_neighbors=20)
second_knn.fit(X_second_knn)

# Find the indices of nearest neighbors to the input song using the second KNN
input_song_neighbors_second = second_knn.kneighbors([X_second_knn.iloc[0]], return_distance=False)[0]

# Recommended Songs

# Get the indices of the top 20 songs nearest to the input song
top_20_song_indices = input_song_neighbors_second

# Retrieve the names of the top 20 songs
print("Names of the top 20 songs nearest to", input_song_name, ":")
for i, song_index in enumerate(top_20_song_indices):
    song_name = df.iloc[song_index]['track_name']
    print(i+1, song_name)



# Both layers of KNN but built from scratch

import numpy as np

# Define the KNN function
def knn(X, query_point, k=5):
    distances = []
    # Calculate the Euclidean distance between the query point and each point in X
    for i in range(len(X)):
        distance = np.linalg.norm(X[i] - query_point)
        distances.append((i, distance))
    # Sort the distances and get the indices of the k nearest neighbors
    sorted_distances = sorted(distances, key=lambda x: x[1])
    nearest_neighbors_indices = [x[0] for x in sorted_distances[:k]]
    return nearest_neighbors_indices

# Define the columns for the first KNN
first_knn_columns = ['dating', 'violence', 'world/life', 'night/time', 'shake the audience', 'family/gospel',
                     'romantic', 'communication', 'obscene', 'music', 'movement/places', 'light/visual perceptions',
                     'family/spiritual', 'like/girls', 'sadness', 'feelings']

# Extract the features for the first KNN
X_first_knn = df[first_knn_columns].values

# Define the columns for the second KNN
second_knn_columns = ['danceability', 'loudness', 'acousticness', 'instrumentalness', 'valence', 'energy']

# Extract the features for the second KNN
X_second_knn = df[second_knn_columns].values

# Function to find nearest neighbors using KNN
def find_nearest_neighbors(X, query_point, k=5):
    distances = []
    # Calculate the Euclidean distance between the query point and each point in X
    for i in range(len(X)):
        distance = np.linalg.norm(X[i] - query_point)
        distances.append((i, distance))
    # Sort the distances and get the indices of the k nearest neighbors
    sorted_distances = sorted(distances, key=lambda x: x[1])
    nearest_neighbors_indices = [x[0] for x in sorted_distances[:k]]
    return nearest_neighbors_indices

# Find the indices of nearest neighbors to the input song using the first KNN
input_song_index = 4  # Change this to the index of the input song
query_point_first_knn = X_first_knn[input_song_index]
input_song_neighbors_first = knn(X_first_knn, query_point_first_knn, k=100)

# Find the indices of nearest neighbors to the input song using the second KNN
query_point_second_knn = X_second_knn[input_song_neighbors_first[0]]
input_song_neighbors_second = knn(X_second_knn, query_point_second_knn, k=20)

# Retrieve the names of the top 20 songs nearest to the input song
print("Names of the top 20 songs nearest to", input_song_name, ":")
for i, song_index in enumerate(input_song_neighbors_second):
    song_name = df.iloc[song_index]['track_name']
    print(i+1, song_name)
