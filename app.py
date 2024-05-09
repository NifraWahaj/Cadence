from flask import Flask, render_template, request, jsonify
import subprocess
import json

# good years zayn
# beyond the reef marty robbins
app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/recommend", methods=["POST"])
def recommend():
    input_song_name = request.form["song_name"]
    input_artist_name = request.form["artist_name"]
    print("FROM JSON:  ", input_song_name, input_artist_name)

    recommendations = []
    # client_id = "285310898d54476f9b5f27c10f666371"
    # client_secret = "3eae50b7f4ef437b91a08150287493fd"

    # import spotipy
    # from spotipy.oauth2 import SpotifyClientCredentials
    # import json
    # import csv

    # song_names = [
    #     "Shape of You",
    #     "Uptown Funk",
    #     "Old Town Road",
    #     "Blinding Lights",
    #     "Closer",
    #     "Dance Monkey",
    #     "Someone You Loved",
    #     "Despacito",
    #     "Sorry",
    #     "Roar",
    #     "Havana",
    #     "Don't Stop Believin'",
    #     "Hello",
    #     "Believer",
    #     "Shake It Off",
    #     "Happier",
    #     "Bad Guy",
    #     "Rockstar",
    #     "Thinking Out Loud",
    #     "Rolling in the Deep",
    #     "Bohemian Rhapsody",
    #     "Radioactive",
    #     "Counting Stars",
    #     "Watermelon Sugar",
    #     "Sicko Mode",
    #     "Hotline Bling",
    #     "Perfect",
    #     "The Box",
    #     "All of Me",
    #     "Love Yourself",
    #     "Can't Stop the Feeling!",
    #     "Humble",
    #     "Sugar",
    #     "Roses",
    #     "Stay",
    #     "Thunder",
    # ]

    # # Fetching Spotify Track IDs

    # # Spotipy library used to interact with the Spotify API, aiming to retrieve the track
    # # IDs for the list of test songs. it initializes a spotipy client with the provided client ID & secret.
    # # Then, it iterates through each song name in the list, conducting a search on Spotify for
    # # tracks matching each song name. For each search result, it extracts the trackID and appends it to song_ids list.

    # client_credentials_manager = SpotifyClientCredentials(
    #     client_id=client_id, client_secret=client_secret
    # )
    # sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

    # song_ids = []

    # for song_name in song_names:
    #     results = sp.search(q=song_name, type="track", limit=1)
    #     for item in results["tracks"]["items"]:
    #         song_ids.append(item["id"])

    # # Retrieving Detailed Information for Spotify Tracks
    # # spotify API is used to get detailed info about the tracks identified by their trackIDs stored in the
    # # song_ids list. By calling the sp. tracks method on the Spotipy client (sp), the script sends a
    # # request to the Spotify API to fetch details about the specified tracks. the retrieved info includes track names,
    # # artists, albums, release dates, popularity scores, other metadata and is stored in the variable tracks_info.

    # tracks_info = sp.tracks(song_ids)

    # # Retrieving Audio Features for Spotify Tracks
    # # spotify api used to fetch acquire audio features for the tracks by invoking the sp.audio_features method.
    # # it sends a request to retrieve audio feature data for the specified tracks. the fetched audio feature info like attributes
    # # such as acousticness, danceability, energy, instrumentalness, tempo, & others, is stored in the variable tracks_audio_features.
    # tracks_audio_features = sp.audio_features(song_ids)

    # # Saving Spotify Tracks Info to JSON Files
    # # the retrieved info about tracks and their audio features is saved to separate JSON files.
    # # json.dump method is used to serialize the dictionaries (tracks_info & tracks_audio_features) containing
    # # the track info & audio features into JSON format. tracks detailed information stored in 'tracks_info.json'
    # # audio featuresstored in 'tracks_features.json'.

    # with open("tracks_info.json", "w") as outfile:
    #     json.dump(tracks_info, outfile, indent=4)
    # with open("tracks_features.json", "w") as outfile:
    #     json.dump(tracks_audio_features, outfile, indent=4)

    # # Data Preprocessing
    # # Processing Spotify Tracks Information and Saving to CSV
    # # track info, initially stored in a JSON file named tracks_info.json, into a structured CSV
    # # format for enhanced accessibility and analysis. The process commences with the retrieval of the tracks's
    # # information from the JSON file, facilitated by opening the file and loading its contents into a Python dictionary
    # # named tracks_info. Subsequently, the script iterates through each track within this dictionary, meticulously
    # # extracting pertinent details such as album information, artist details, track duration, popularity, and more.
    # # To enrich the dataset, a function get_album_genres is introduced to obtain album genres based on the album ID.
    # # For each track, a dictionary named track_detail is meticulously constructed, encapsulating
    # # all the extracted information, including the fetched album genres. Following this data refinement,
    # # the script proceeds to organize the track details into a CSV file, denoted as track_details.csv, with each row
    # # representing a distinct track and its associated attributes aligned with the predefined field names.
    # # Finally, a confirmation message is printed, affirming the successful completion of the process and the
    # # saving of track details into the CSV file. This meticulous conversion process ensures the seamless transition of
    # # Spotify tracks' information into a structured format conducive to comprehensive analysis, visualization, and integration
    # # within diverse analytical workflows or applications.

    # with open("tracks_info.json", "r") as file:
    #     tracks_info = json.load(file)

    # track_details = []

    # def get_album_genres(album_id):
    #     album_info = sp.album(album_id)
    #     if "genres" in album_info:
    #         return album_info["genres"]
    #     else:
    #         return []

    # for track in tracks_info["tracks"]:
    #     track_detail = {
    #         "album_id": track["album"]["id"],
    #         "album_name": track["album"]["name"],
    #         "album_release_date": track["album"]["release_date"],
    #         "album_total_tracks": track["album"]["total_tracks"],
    #         "album_type": track["album"]["album_type"],
    #         "artist_id": track["artists"][0]["id"],
    #         "artist_name": track["artists"][0]["name"],
    #         "artist_type": track["artists"][0]["type"],
    #         "available_markets": track["available_markets"],
    #         "disc_number": track["disc_number"],
    #         "duration_ms": track["duration_ms"],
    #         "explicit": track["explicit"],
    #         "isrc": track["external_ids"]["isrc"] if "external_ids" in track else None,
    #         "id": track["id"],
    #         "is_local": track["is_local"],
    #         "name": track["name"],
    #         "popularity": track["popularity"],
    #         "track_number": track["track_number"],
    #     }

    #     album_id = track_detail["album_id"]
    #     album_genres = get_album_genres(album_id)
    #     track_detail["album_genres"] = album_genres
    #     track_details.append(track_detail)

    # fieldnames = [
    #     "album_id",
    #     "album_name",
    #     "album_release_date",
    #     "album_total_tracks",
    #     "album_type",
    #     "artist_id",
    #     "artist_name",
    #     "artist_type",
    #     "available_markets",
    #     "disc_number",
    #     "duration_ms",
    #     "explicit",
    #     "isrc",
    #     "id",
    #     "is_local",
    #     "name",
    #     "popularity",
    #     "track_number",
    #     "album_genres",
    # ]

    # with open("track_details.csv", "w", newline="") as csvfile:
    #     writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    #     writer.writeheader()
    #     for track_detail in track_details:
    #         writer.writerow(track_detail)

    # print("Track details have been saved to track_details.csv")

    # # Processing Spotify Tracks Audio Features and Saving to CSV
    # # processing the audio features of Spotify tracks, initially stored in a JSON file named tracks_features.json,
    # # and saving them into a structured CSV format for further analysis or integration. Firstly, the script reads
    # # the contents of the tracks_features.json file, which contains the audio features of the Spotify tracks, and
    # # loads them into a Python dictionary named tracks_features. Next, it iterates through each track in the
    # # tracks_features dictionary, selectively extracting essential audio features such as danceability, energy, key,
    # # loudness, mode, speechiness, acousticness, instrumentalness, liveness, valence, tempo, duration, and time signature.
    # # These features are crucial for analyzing the musical characteristics and styles of the tracks comprehensively.
    # # For each track, a dictionary named track_features is constructed, encapsulating the extracted audio features.
    # # These dictionaries collectively form a list named audio_features, which holds the audio features of all the
    # # tracks in a structured manner. Subsequently, the script organizes the track features into a CSV file named
    # # track_audio_features.csv. It defines the field names for the CSV file based on the extracted audio features
    # # and utilizes the csv.DictWriter class to write the audio features into the CSV file. Each row in the CSV file
    # # represents a distinct track, with its associated audio features aligned with the predefined field names

    # import csv

    # with open("tracks_features.json", "r") as file:
    #     tracks_features = json.load(file)

    # audio_features = []

    # for track in tracks_features:
    #     track_features = {
    #         "danceability": track["danceability"],
    #         "energy": track["energy"],
    #         "key": track["key"],
    #         "loudness": track["loudness"],
    #         "mode": track["mode"],
    #         "speechiness": track["speechiness"],
    #         "acousticness": track["acousticness"],
    #         "instrumentalness": track["instrumentalness"],
    #         "liveness": track["liveness"],
    #         "valence": track["valence"],
    #         "tempo": track["tempo"],
    #         "id": track["id"],
    #         "duration_ms": track["duration_ms"],
    #         "time_signature": track["time_signature"],
    #     }
    #     audio_features.append(track_features)

    # fieldnames = [
    #     "danceability",
    #     "energy",
    #     "key",
    #     "loudness",
    #     "mode",
    #     "speechiness",
    #     "acousticness",
    #     "instrumentalness",
    #     "liveness",
    #     "valence",
    #     "tempo",
    #     "id",
    #     "duration_ms",
    #     "time_signature",
    # ]

    # with open("track_audio_features.csv", "w", newline="") as csvfile:
    #     writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    #     writer.writeheader()
    #     for track_feature in audio_features:
    #         writer.writerow(track_feature)

    # print("Audio features have been saved to track_audio_features.csv")

    # # Merging Track Details and Audio Features Data and Saving to CSV
    # # combining info about Spotify tracks' details and their audio features into a single dataset.
    # # First, it reads the track details (like album, artist, etc.) and audio features (like danceability, energy, etc.)
    # # from csv files. Then, it goes through each track and its corresponding audio features, matching them based
    # # on their track IDs. When a match is found, it merges the details and features into one entry, creating a
    # # combined dataset.
    # import csv

    # track_details = []
    # with open("track_details.csv", "r", newline="") as csvfile:
    #     reader = csv.DictReader(csvfile)
    #     for row in reader:
    #         track_details.append(row)

    # audio_features = []
    # with open("track_audio_features.csv", "r", newline="") as csvfile:
    #     reader = csv.DictReader(csvfile)
    #     for row in reader:
    #         audio_features.append(row)

    # merged_data = []
    # for detail in track_details:
    #     for feature in audio_features:
    #         if detail["id"] == feature["id"]:
    #             merged_data.append({**detail, **feature})
    #             break

    # fieldnames = [
    #     "album_id",
    #     "album_name",
    #     "album_release_date",
    #     "album_total_tracks",
    #     "album_type",
    #     "artist_id",
    #     "artist_name",
    #     "artist_type",
    #     "available_markets",
    #     "disc_number",
    #     "duration_ms",
    #     "explicit",
    #     "isrc",
    #     "is_local",
    #     "name",
    #     "popularity",
    #     "track_number",
    #     "album_genres",
    #     "danceability",
    #     "energy",
    #     "key",
    #     "loudness",
    #     "mode",
    #     "speechiness",
    #     "acousticness",
    #     "instrumentalness",
    #     "liveness",
    #     "valence",
    #     "tempo",
    #     "id",
    #     "time_signature",
    # ]

    # with open("merged_track_data.csv", "w", newline="") as csvfile:
    #     writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    #     writer.writeheader()
    #     for item in merged_data:
    #         writer.writerow(item)

    # print("Merged track data has been saved to merged_track_data.csv")

    # Feature Engineering
    # Reading CSV File and Displaying Missing Values using Pandas
    # uses pandas library to read "songs.csv" and displays the count of missing values in each column of the DF.
    # pd.read_csv method used to to read the file & load contents into DF 'df'. then, calculating the number of
    # missing values (NaN or NULL) present in each column. the sum of missing values for each column calculating.
    # printing missing_values series, which displays the count of missing values for each column in the DF.
    # -----------------------------------
    # import pandas as pd

    # df = pd.read_csv("songs.csv")
    # missing_values = df.isnull().sum()
    # print("missing_values", missing_values)

    # from sklearn.preprocessing import MinMaxScaler, LabelEncoder

    # df["song"] = df["track_name"] + df["artist_name"]

    # df.to_csv("output.csv", index=False)

    # df.drop(["lyrics", "len"], axis=1, inplace=True)

    # df = pd.get_dummies(df, columns=["topic", "artist_name", "genre"])

    # scaler = MinMaxScaler()
    # df["release_date"] = scaler.fit_transform(df[["release_date"]])

    # print("df.head()", df.head())

    # all_columns = df.columns

    # topic_columns = [col for col in all_columns if col.startswith("topic")]

    # all_columns = df.columns
    # topic_columns = [col for col in all_columns if col.startswith("genre")]

    # print("all_columns", all_columns)
    # print("topic_columns",topic_columns)
    # import numpy as np

    # def custom_similarity(song1, song2):
    #     features_song1 = np.array(
    #         [song1["age"]] + [song1[col] for col in topic_genre_columns]
    #     )
    #     features_song2 = np.array(
    #         [song2["age"]] + [song2[col] for col in topic_genre_columns]
    #     )
    #     # euclidean dist b/w feature vectors
    #     distance = np.linalg.norm(features_song1 - features_song2)

    #     return distance

    # topic_genre_columns = [
    #     col for col in df.columns if col.startswith("topic") or col.startswith("genre")
    # ]

    # def find_song_index(song_name):
    #     return df.index[df["song"] == song_name].tolist()

    # input_song_name = input_song_name + input_artist_name

    # # Find the index of the input song
    # input_song_index = find_song_index(input_song_name)
    # # print("input_song_name ", input_song_name)
    # # print("input_song_index ", input_song_index)
    # if not input_song_index:
    #     print("Song not found.")
    #     recommendations = []
    #     recommendations.append({"Song not found.": song_name})
    #     exit()
    # # print(input_song_index)
    # input_song_index = input_song_index[
    #     0
    # ]  # If multiple occurrences found, take the first one

    # # Get features of the input song
    # input_song_features = df.iloc[input_song_index]

    # # Calculate similarity scores between input song and all other songs
    # similarity_scores = [
    #     (i, custom_similarity(input_song_features, df.iloc[i])) for i in range(len(df))
    # ]

    # # Sort songs based on similarity scores
    # sorted_songs = sorted(similarity_scores, key=lambda x: x[1])

    # # Print the first 10 elements of similarity_scores
    # print("First 10 elements of similarity_scores:")
    # print(similarity_scores[:10])

    # # Print the last 10 elements of similarity_scores
    # print("Last 10 elements of similarity_scores:")
    # print(similarity_scores[-10:])

    # # Print the first 10 elements of sorted_songs
    # print("First 10 elements of sorted_songs:")
    # print(sorted_songs[:10])

    # # Print the last 10 elements of sorted_songs
    # print("Last 10 elements of sorted_songs:")
    # print(sorted_songs[-10:])

    # # Select the top 2000 most similar songs
    # top_2000_songs = sorted_songs[:2000]
    # print("First 10 elements of top_2000_songs:")
    # print(top_2000_songs[:10])

    # # Print the last 10 elements of sorted_songs
    # print("Last 10 elements of top_2000_songs:")
    # print(top_2000_songs[-10:])
    # # print(top_2000_songs)

    import pandas as pd
    import numpy as np
    from sklearn.preprocessing import MinMaxScaler

    # Load the songs dataset
    df = pd.read_csv("songs.csv")

    # Check for missing values
    missing_values = df.isnull().sum()
    print("Missing values:", missing_values)

    # Combine 'track_name' and 'artist_name' to create a 'song' column
    df["song"] = df["track_name"] + " - " + df["artist_name"]

    artist_name_song_name = df["song"]
    print("DF[SONG]", df["song"])
    # Save the modified DataFrame to a new CSV file
    df.to_csv("output.csv", index=False)

    # Drop unnecessary columns
    df.drop(["lyrics", "len"], axis=1, inplace=True)

    # One-hot encode categorical columns ('topic', 'artist_name', 'genre')
    df = pd.get_dummies(df, columns=["topic", "artist_name", "genre"])

    # Normalize 'release_date' using Min-Max scaling
    scaler = MinMaxScaler()
    df["release_date"] = scaler.fit_transform(df[["release_date"]])

    # Display the first few rows of the modified DataFrame
    print("df.head():", df.head())

    # Define columns related to 'topic' and 'genre'
    topic_genre_columns = [
        col for col in df.columns if col.startswith("topic") or col.startswith("genre")
    ]

    # Function to calculate custom similarity between two songs
    def custom_similarity(song1, song2):
        features_song1 = np.array(
            [song1["release_date"]] + [song1[col] for col in topic_genre_columns]
        )
        features_song2 = np.array(
            [song2["release_date"]] + [song2[col] for col in topic_genre_columns]
        )
        # Calculate Euclidean distance between feature vectors
        distance = np.linalg.norm(features_song1 - features_song2)
        return distance

    # Function to find index of a song in DataFrame
    def find_song_index(song_name):
        print("SONG NAME: ", song_name)
        return df.index[df["song"] == song_name].tolist()

    input_song_name = input_song_name + " - " + input_artist_name
    print("input_song_name: ", input_song_name)
    # Find the index of the input song
    input_song_index = find_song_index(input_song_name)

    print("input_song_index", input_song_index)
    if not input_song_index:
        print("Song not found.")
        exit()

    input_song_index = input_song_index[
        0
    ]  # Take the first occurrence if multiple found

    # Get features of the input song
    input_song_features = df.iloc[input_song_index]
    print("input_song_features", input_song_features)

    # Assuming input_song_features is a DataFrame containing the features of a specific song
    # Extract the value from the "Unnamed: 0" column of input_song_features
    song_index_value_self = input_song_features.loc["Unnamed: 0"]

    # Print the extracted value
    print("Song Index Value:", song_index_value_self)

    # # Calculate similarity scores between input song and all other songs
    # similarity_scores = [
    #     (
    #         df.iloc[i]["song"],
    #         custom_similarity(input_song_features, df.iloc[i]),
    #     )
    #     for i in range(len(df))
    # ]
    # Calculate similarity scores between input song and all other songs
    # for i in range(len(df)):
    #     similarity_scores.append({"song_name": "-------"})

    # similarity_scores = [
    #     (i, custom_similarity(input_song_features, df.iloc[i])) for i in range(len(df))
    # ]

    # Initialize an empty list to store similarity scores
    similarity_scores = []

    # Iterate over each row in the DataFrame
    for i in range(len(df)):
        # Get the song name for the current row
        song_name = df.iloc[i]["song"]

        # Calculate the similarity score for the current song
        similarity_score = custom_similarity(input_song_features, df.iloc[i])

        # Append a tuple of (song_name, similarity_score) to similarity_scores list
        similarity_scores.append((song_name, similarity_score))

        # Print values for debugging (optional)
    # print(f"i: {i}, song: {song_name}, similarity: {similarity_score}")

    print("\n\nFirst 10 elements of similarity_scores:")
    print(similarity_scores[:10])
    print("\n\nLast 10 elements of similarity_scores:")
    print(similarity_scores[-10:])

    sorted_similarity_scores = sorted(similarity_scores, key=lambda x: x[1])
    print("\n\nFirst 10 elements of sorted_similarity_scores:")
    print(sorted_similarity_scores[:10])
    print("\n\nLast 10 elements of sorted_similarity_scores:")
    print(sorted_similarity_scores[-10:])

    recommended_songs = [
        song_name for song_name, similarity_score in sorted_similarity_scores
    ]

    # print("FIRST RECOMMENDATION ")
    # # Print the recommended song names
    # for song_name in recommended_songs:
    #     recommendations.append({"song_name": song_name})
    #     print("song_name", song_name)
    # recommendations.append({"song_name": "-------"})

    # Define the maximum number of top songs to include
    top_count = 20

    # Iterate over recommended_songs and add top songs to recommendations
    for index, song_name in enumerate(recommended_songs):
        # Add each song to recommendations dictionary
        recommendations.append({"song_name": song_name})

        # Print song_name for debugging (optional)
        print("song_name:", song_name)

        # Check if we have added the maximum number of top songs
        if index + 1 >= top_count:
            break  # Exit loop after adding the top_count songs

    # Add a separator after the top songs
    recommendations.append({"song_name": "-------"})

    # Sort songs based on similarity scores
    # sorted_songs = sorted(similarity_scores, key=lambda x: x[2])

    # Print the first 10 and last 10 elements of similarity_scores
    # print("First 10 elements of similarity_scores:")
    # print(sorted_songs[:10])
    # print("Last 10 elements of similarity_scores:")
    # print(sorted_songs[-10:])

    # # Select the top 2000 most similar songs
    # top_2000_songs = sorted_songs[:2000]

    # # Print the first 10 and last 10 elements of top_2000_songs
    # print("First 10 elements of top_2000_songs:")
    # print(top_2000_songs[:10])
    # print("Last 10 elements of top_2000_songs:")
    # print(top_2000_songs[-10:])

    from sklearn.neighbors import NearestNeighbors

    # Define the columns for the first KNN
    first_knn_columns = [
        "dating",
        "violence",
        "world/life",
        "night/time",
        "shake the audience",
        "family/gospel",
        "romantic",
        "communication",
        "obscene",
        "music",
        "movement/places",
        "light/visual perceptions",
        "family/spiritual",
        "like/girls",
        "sadness",
        "feelings",
    ]

    # Extract the features for the first KNN
    X_first_knn = df[first_knn_columns]
    # print("First 10 elements of X_first_knn:")
    # print(X_first_knn[:10])

    # # Print the last 10 elements of sorted_songs
    # print("Last 10 elements of X_first_knn:")
    # print(X_first_knn[-10:])
    # print("X_first_knn: ", X_first_knn)

    # Instantiate the first KNN model
    first_knn = NearestNeighbors(n_neighbors=100)
    first_knn.fit(X_first_knn)

    # Get indices of the first 10 nearest neighbors for the first sample
    first_sample = X_first_knn.iloc[0].values.reshape(
        1, -1
    )  # Reshape for single sample
    distances, indices = first_knn.kneighbors(first_sample)

    # Extract the song names and distances of the first 10 nearest neighbors
    nearest_neighbors = df.iloc[indices[0][:10]]
    nearest_song_names = nearest_neighbors["song"].tolist()
    nearest_distances = distances[0][:10]

    # Print the first 10 elements of first_knn with song names and distances
    print("First 10 elements of first_knn (with song names and distances):")
    for song_name, distance in zip(nearest_song_names, nearest_distances):
        print(
            f"Song: {song_name}, Distance: {distance:.4f}"
        )  # print("First 10 elements of first_knn:")
    # print(first_knn[:10])

    # # Print the last 10 elements of sorted_songs
    # print("Last 10 elements of first_knn:")
    # print(first_knn[-10:])

    # Find the indices of nearest neighbors to the input song using the first KNN
    input_song_neighbors_first = first_knn.kneighbors(
        [X_first_knn.iloc[input_song_index]], return_distance=False
    )[0]

    # Second KNN layer to prioritize songs based upon the musical factors

    # Define the columns for the second KNN
    second_knn_columns = [
        "danceability",
        "loudness",
        "acousticness",
        "instrumentalness",
        "valence",
        "energy",
    ]

    # Extract the features for the second KNN
    X_second_knn = df.iloc[input_song_neighbors_first][second_knn_columns]

    # Instantiate the second KNN model
    second_knn = NearestNeighbors(n_neighbors=20)
    second_knn.fit(X_second_knn)

    # Find the indices of nearest neighbors to the input song using the second KNN
    input_song_neighbors_second = second_knn.kneighbors(
        [X_second_knn.iloc[0]], return_distance=False
    )[0]

    # Recommended Songs

    # Get the indices of the top 20 songs nearest to the input song
    top_20_song_indices = input_song_neighbors_second

    # Retrieve the names of the top 20 songs

    print("Names of the top 20 songs nearest to", input_song_name, ":")
    for i, song_index in enumerate(top_20_song_indices):
        song_name = df.iloc[song_index]["track_name"]
        recommendations.append({"song_name": song_name})
        print(i + 1, song_name)

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
    first_knn_columns = [
        "dating",
        "violence",
        "world/life",
        "night/time",
        "shake the audience",
        "family/gospel",
        "romantic",
        "communication",
        "obscene",
        "music",
        "movement/places",
        "light/visual perceptions",
        "family/spiritual",
        "like/girls",
        "sadness",
        "feelings",
    ]

    # Extract the features for the first KNN
    X_first_knn = df[first_knn_columns].values

    # Define the columns for the second KNN
    second_knn_columns = [
        "danceability",
        "loudness",
        "acousticness",
        "instrumentalness",
        "valence",
        "energy",
    ]

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
    # input_song_index = 4  # Change this to the index of the input song
    query_point_first_knn = X_first_knn[input_song_index]
    input_song_neighbors_first = knn(X_first_knn, query_point_first_knn, k=100)

    # Find the indices of nearest neighbors to the input song using the second KNN
    query_point_second_knn = X_second_knn[input_song_neighbors_first[0]]
    input_song_neighbors_second = knn(X_second_knn, query_point_second_knn, k=20)
    print(df)

    for i, song_index in enumerate(input_song_neighbors_second):
        song_name = df.iloc[song_index]["track_name"]
        artist_name = df.iloc[song_index]["track_name"]
        recommendations.append({"song_name": song_name})
        print(i + 1, song_name, artist_name)

    print("DONE")
    song_name = ""
    artist_name = ""
    # Return recommendations as JSON response
    return jsonify({"recommendations": recommendations})

    # ===============================================================================================================
    # Parse the output JSON
    # recommendations = json.loads(output.decode("utf-8"))

    # Print the recommendations
    # print("Recommendations:")
    ##for recommendation in recommendations:P
    #    print(recommendation)

    # Return the recommendations as JSON response


# return jsonify({'recommendations': recommendations})

if __name__ == "__main__":
    app.run(debug=True)
