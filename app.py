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
        "Thunder",
    ]

    # Fetching Spotify Track IDs

    # Spotipy library used to interact with the Spotify API, aiming to retrieve the track
    # IDs for the list of test songs. it initializes a spotipy client with the provided client ID & secret.
    # Then, it iterates through each song name in the list, conducting a search on Spotify for
    # tracks matching each song name. For each search result, it extracts the trackID and appends it to song_ids list.

    client_credentials_manager = SpotifyClientCredentials(
        client_id=client_id, client_secret=client_secret
    )
    sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

    song_ids = []

    for song_name in song_names:
        results = sp.search(q=song_name, type="track", limit=1)
        for item in results["tracks"]["items"]:
            song_ids.append(item["id"])

    # Retrieving Detailed Information for Spotify Tracks
    # spotify API is used to get detailed info about the tracks identified by their trackIDs stored in the
    # song_ids list. By calling the sp. tracks method on the Spotipy client (sp), the script sends a
    # request to the Spotify API to fetch details about the specified tracks. the retrieved info includes track names,
    # artists, albums, release dates, popularity scores, other metadata and is stored in the variable tracks_info.

    tracks_info = sp.tracks(song_ids)

    # Retrieving Audio Features for Spotify Tracks
    # spotify api used to fetch acquire audio features for the tracks by invoking the sp.audio_features method.
    # it sends a request to retrieve audio feature data for the specified tracks. the fetched audio feature info like attributes
    # such as acousticness, danceability, energy, instrumentalness, tempo, & others, is stored in the variable tracks_audio_features.
    tracks_audio_features = sp.audio_features(song_ids)

    # Saving Spotify Tracks Info to JSON Files
    # the retrieved info about tracks and their audio features is saved to separate JSON files.
    # json.dump method is used to serialize the dictionaries (tracks_info & tracks_audio_features) containing
    # the track info & audio features into JSON format. tracks detailed information stored in 'tracks_info.json'
    # audio featuresstored in 'tracks_features.json'.

    with open("tracks_info.json", "w") as outfile:
        json.dump(tracks_info, outfile, indent=4)
    with open("tracks_features.json", "w") as outfile:
        json.dump(tracks_audio_features, outfile, indent=4)

    # Data Preprocessing
    # Processing Spotify Tracks Information and Saving to CSV
    # track info, initially stored in a JSON file named tracks_info.json, into a structured CSV
    # format for enhanced accessibility and analysis. The process commences with the retrieval of the tracks's
    # information from the JSON file, facilitated by opening the file and loading its contents into a Python dictionary
    # named tracks_info. Subsequently, the script iterates through each track within this dictionary, meticulously
    # extracting pertinent details such as album information, artist details, track duration, popularity, and more.
    # To enrich the dataset, a function get_album_genres is introduced to obtain album genres based on the album ID.
    # For each track, a dictionary named track_detail is meticulously constructed, encapsulating
    # all the extracted information, including the fetched album genres. Following this data refinement,
    # the script proceeds to organize the track details into a CSV file, denoted as track_details.csv, with each row
    # representing a distinct track and its associated attributes aligned with the predefined field names.
    # Finally, a confirmation message is printed, affirming the successful completion of the process and the
    # saving of track details into the CSV file. This meticulous conversion process ensures the seamless transition of
    # Spotify tracks' information into a structured format conducive to comprehensive analysis, visualization, and integration
    # within diverse analytical workflows or applications.

    with open("tracks_info.json", "r") as file:
        tracks_info = json.load(file)

    track_details = []

    def get_album_genres(album_id):
        album_info = sp.album(album_id)
        if "genres" in album_info:
            return album_info["genres"]
        else:
            return []

    for track in tracks_info["tracks"]:
        track_detail = {
            "album_id": track["album"]["id"],
            "album_name": track["album"]["name"],
            "album_release_date": track["album"]["release_date"],
            "album_total_tracks": track["album"]["total_tracks"],
            "album_type": track["album"]["album_type"],
            "artist_id": track["artists"][0]["id"],
            "artist_name": track["artists"][0]["name"],
            "artist_type": track["artists"][0]["type"],
            "available_markets": track["available_markets"],
            "disc_number": track["disc_number"],
            "duration_ms": track["duration_ms"],
            "explicit": track["explicit"],
            "isrc": track["external_ids"]["isrc"] if "external_ids" in track else None,
            "id": track["id"],
            "is_local": track["is_local"],
            "name": track["name"],
            "popularity": track["popularity"],
            "track_number": track["track_number"],
        }

        album_id = track_detail["album_id"]
        album_genres = get_album_genres(album_id)
        track_detail["album_genres"] = album_genres
        track_details.append(track_detail)

    fieldnames = [
        "album_id",
        "album_name",
        "album_release_date",
        "album_total_tracks",
        "album_type",
        "artist_id",
        "artist_name",
        "artist_type",
        "available_markets",
        "disc_number",
        "duration_ms",
        "explicit",
        "isrc",
        "id",
        "is_local",
        "name",
        "popularity",
        "track_number",
        "album_genres",
    ]

    with open("track_details.csv", "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for track_detail in track_details:
            writer.writerow(track_detail)

    print("Track details have been saved to track_details.csv")

    # Processing Spotify Tracks Audio Features and Saving to CSV
    # processing the audio features of Spotify tracks, initially stored in a JSON file named tracks_features.json,
    # and saving them into a structured CSV format for further analysis or integration. Firstly, the script reads
    # the contents of the tracks_features.json file, which contains the audio features of the Spotify tracks, and
    # loads them into a Python dictionary named tracks_features. Next, it iterates through each track in the
    # tracks_features dictionary, selectively extracting essential audio features such as danceability, energy, key,
    # loudness, mode, speechiness, acousticness, instrumentalness, liveness, valence, tempo, duration, and time signature.
    # These features are crucial for analyzing the musical characteristics and styles of the tracks comprehensively.
    # For each track, a dictionary named track_features is constructed, encapsulating the extracted audio features.
    # These dictionaries collectively form a list named audio_features, which holds the audio features of all the
    # tracks in a structured manner. Subsequently, the script organizes the track features into a CSV file named
    # track_audio_features.csv. It defines the field names for the CSV file based on the extracted audio features
    # and utilizes the csv.DictWriter class to write the audio features into the CSV file. Each row in the CSV file
    # represents a distinct track, with its associated audio features aligned with the predefined field names

    import csv

    with open("tracks_features.json", "r") as file:
        tracks_features = json.load(file)

    audio_features = []

    for track in tracks_features:
        track_features = {
            "danceability": track["danceability"],
            "energy": track["energy"],
            "key": track["key"],
            "loudness": track["loudness"],
            "mode": track["mode"],
            "speechiness": track["speechiness"],
            "acousticness": track["acousticness"],
            "instrumentalness": track["instrumentalness"],
            "liveness": track["liveness"],
            "valence": track["valence"],
            "tempo": track["tempo"],
            "id": track["id"],
            "duration_ms": track["duration_ms"],
            "time_signature": track["time_signature"],
        }
        audio_features.append(track_features)

    fieldnames = [
        "danceability",
        "energy",
        "key",
        "loudness",
        "mode",
        "speechiness",
        "acousticness",
        "instrumentalness",
        "liveness",
        "valence",
        "tempo",
        "id",
        "duration_ms",
        "time_signature",
    ]

    with open("track_audio_features.csv", "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for track_feature in audio_features:
            writer.writerow(track_feature)

    print("Audio features have been saved to track_audio_features.csv")

    # Merging Track Details and Audio Features Data and Saving to CSV
    # combining info about Spotify tracks' details and their audio features into a single dataset.
    # First, it reads the track details (like album, artist, etc.) and audio features (like danceability, energy, etc.)
    # from csv files. Then, it goes through each track and its corresponding audio features, matching them based
    # on their track IDs. When a match is found, it merges the details and features into one entry, creating a
    # combined dataset.
    import csv

    track_details = []
    with open("track_details.csv", "r", newline="") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            track_details.append(row)

    audio_features = []
    with open("track_audio_features.csv", "r", newline="") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            audio_features.append(row)

    merged_data = []
    for detail in track_details:
        for feature in audio_features:
            if detail["id"] == feature["id"]:
                merged_data.append({**detail, **feature})
                break

    fieldnames = [
        "album_id",
        "album_name",
        "album_release_date",
        "album_total_tracks",
        "album_type",
        "artist_id",
        "artist_name",
        "artist_type",
        "available_markets",
        "disc_number",
        "duration_ms",
        "explicit",
        "isrc",
        "is_local",
        "name",
        "popularity",
        "track_number",
        "album_genres",
        "danceability",
        "energy",
        "key",
        "loudness",
        "mode",
        "speechiness",
        "acousticness",
        "instrumentalness",
        "liveness",
        "valence",
        "tempo",
        "id",
        "time_signature",
    ]

    with open("merged_track_data.csv", "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for item in merged_data:
            writer.writerow(item)

    print("Merged track data has been saved to merged_track_data.csv")

    # Feature Engineering
    # Reading CSV File and Displaying Missing Values using Pandas
    # uses pandas library to read "songs.csv" and displays the count of missing values in each column of the DF.
    # pd.read_csv method used to to read the file & load contents into DF 'df'. then, calculating the number of
    # missing values (NaN or NULL) present in each column. the sum of missing values for each column calculating.
    # printing missing_values series, which displays the count of missing values for each column in the DF.
    import pandas as pd

    df = pd.read_csv("songs.csv")
    missing_values = df.isnull().sum()
    print(missing_values)

    # Data Preprocessing with Min-Max Normalization, One-Hot Encoding, and Feature Engineering
    # using scikit-learn an&d pandas libraries
    # Combining 'track_name' & 'artist_name' into a new col named 'song'. Removing 'lyrics' and 'len' cols from DF.
    # one-hot encoding on categorical columns ('topic', 'artist_name', and 'genre') using the pd.get_dummies function.
    # to convert them into binary vectors. Min-Max Normalization to the 'release_date' col - scales the values to a
    # range between 0 and 1. printing the first few rows of the modified DF
    from sklearn.preprocessing import MinMaxScaler, LabelEncoder

    df["song"] = df["track_name"] + df["artist_name"]

    df.to_csv("output.csv", index=False)

    df.drop(["lyrics", "len"], axis=1, inplace=True)

    df = pd.get_dummies(df, columns=["topic", "artist_name", "genre"])

    scaler = MinMaxScaler()
    df["release_date"] = scaler.fit_transform(df[["release_date"]])

    print(df.head())

    # Filtering Columns Starting with "topic" from the DataFrame
    # initially, it retrieves all col names from df. then, filters out col names starting with the 'topic'.
    # The filtered col names are then stored in topic_columns., which is printed
    all_columns = df.columns

    topic_columns = [col for col in all_columns if col.startswith("topic")]

    # Filtering Columns Starting with "genre" from the DataFrame
    # This cell filters the column names from the DataFrame df, specifically targeting columns that begin with the string "genre".
    # retrieves all col names from df, storing in all_columns. filters out col names that start with 'genre'.

    all_columns = df.columns
    topic_columns = [col for col in all_columns if col.startswith("genre")]

    print(topic_columns)
 ------------------

    def custom_similarity(song1, song2):
        features_song1 = np.array(
            [song1["age"]] + [song1[col] for col in topic_genre_columns]
        )
        features_song2 = np.array(
            [song2["age"]] + [song2[col] for col in topic_genre_columns]
        )
        # euclidean dist b/w feature vectors
        distance = np.linalg.norm(features_song1 - features_song2)

        return distance
    
    topic_genre_columns = [
        col for col in df.columns if col.startswith("topic") or col.startswith("genre")
    ]

    def find_song_index(song_name):
        return df.index[df["song"] == song_name].tolist()

    input_song_name = input_song_name + input_artist_name

    # Find the index of the input song
    input_song_index = find_song_index(input_song_name)
    print("input_song_name ", input_song_name)
    print("input_song_index ", input_song_index)
    if not input_song_index:
        print("Song not found.")
        recommendations = []
        recommendations.append({"Song not found.": song_name})
        exit()
    print(input_song_index)
    input_song_index = input_song_index[
        0
    ]  # If multiple occurrences found, take the first one

    # Get features of the input song
    input_song_features = df.iloc[input_song_index]

    # Calculate similarity scores between input song and all other songs
    similarity_scores = [
        (i, custom_similarity(input_song_features, df.iloc[i])) for i in range(len(df))
    ]

    # Sort songs based on similarity scores
    sorted_songs = sorted(similarity_scores, key=lambda x: x[1])

    # Select the top 2000 most similar songs
    top_2000_songs = sorted_songs[:2000]

    print(top_2000_songs)

    # First KNN layer to prioritize songs based upon context

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

    # Instantiate the first KNN model
    first_knn = NearestNeighbors(n_neighbors=100)
    first_knn.fit(X_first_knn)

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

    recommendations = []

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
        print(i + 1, song_name)

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
