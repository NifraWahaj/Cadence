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
    client_id = ""
    client_secret = ""

    import spotipy
    from spotipy.oauth2 import SpotifyClientCredentials
    import json
    import csv

    # test list of songs
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
    # tracks matching each song name. For each search result, it extracts the trackID and
    # appends it to song_ids list.

    song_ids = []
    client_credentials_manager = SpotifyClientCredentials(
        client_id=client_id, client_secret=client_secret
    )
    sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

    for song_name in song_names:
        results = sp.search(q=song_name, type="track", limit=1)
        for item in results["tracks"]["items"]:
            song_ids.append(item["id"])

    # Retrieving Detailed Information for Spotify Tracks
    # get detailed info about the tracks identified by their trackIDs stored in the song_ids list.
    tracks_info = sp.tracks(song_ids)

    # Retrieving Audio Features for Spotify Tracks
    tracks_audio_features = sp.audio_features(song_ids)

    # # Saving Spotify Tracks Info to JSON Files
    # the retrieved info about tracks and their audio features is saved to separate
    # JSON files. json.dump method is used to serialize the dictionaries (tracks_info & tracks_audio_features)
    # containing the track info & audio features into JSON format. tracks detailed information stored in
    # 'tracks_info.json' audio featuresstored in 'tracks_features.json'.

    with open("tracks_info.json", "w") as outfile:
        json.dump(tracks_info, outfile, indent=4)
    with open("tracks_features.json", "w") as outfile:
        json.dump(tracks_audio_features, outfile, indent=4)

    # Data Preprocessing
    # Processing Spotify Tracks Information and Saving to CSV
    # track info, initially stored in a JSON file named tracks_info.json,
    # into a structured CSV format for enhanced accessibility and analysis.
    # retrieval of the tracks' information from the JSON file, the script iterates through
    # each track, extracting pertinent details . the script proceeds to organize the
    # track details into track_details.csv

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

        # adding album genres
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

    # # Processing Spotify Tracks Audio Features and Saving to CSV
    # processing the audio features of Spotify tracks, initially stored in a JSON
    # file named tracks_features.json, and saving them into a structured CSV format.
    # it iterates through each track in the tracks_features dictionary, selectively
    # extracting essential audio features such as danceability, energy, key, loudness, mode,
    # speechiness, acousticness, instrumentalness, liveness, valence, tempo, duration, and time signature.
    # For each track, a dictionary named track_features is constructed, encapsulating the extracted audio features.
    # the script organizes the track features into a CSV file named track_audio_features.csv. It defines
    # the field names for the CSV file based on the extracted audio features and utilizes the csv.DictWriter
    # class to write the audio features into the CSV file.
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
    # isnull().sum() function call is applied to the entire DataFrame df, which means it
    # calculates the sum of missing values for each column individually

    import pandas as pd
    import numpy as np
    from sklearn.preprocessing import MinMaxScaler

    df = pd.read_csv("songs.csv")
    missing_values = df.isnull().sum()
    # print("Missing values:", missing_values)

    # # Data Preprocessing with Min-Max Normalization, One-Hot Encoding, and Feature Engineering
    # creating a new column named "song" in the df, which concatenates the values of the "track_name"
    # column and the "artist_name" column for each row.

    df["song"] = df["track_name"] + " - " + df["artist_name"]
    df.to_csv("output.csv", index=False)

    # drops the columns 'lyrics' and 'len' from the df. The axis=1 argument specifies that the operation
    # should be performed column-wise, and inplace=True means the changes are made directly to the df
    df.drop(["lyrics", "len"], axis=1, inplace=True)

    # Applying one-hot encoding to the categorical columns 'topic', 'artist_name', and 'genre'.
    # Converting these into dummy/indicator variables.
    df = pd.get_dummies(df, columns=["topic", "artist_name", "genre"])

    # A MinMaxScaler object is created, which will scale 'release_date' to a range between 0 and 1.
    # Then, the column is selected and transformed using fit_transform() method of the scaler.
    # This scales the 'release_date' values to the range between 0 and 1.
    scaler = MinMaxScaler()
    df["release_date"] = scaler.fit_transform(df[["release_date"]])
    # print("df.head():", df.head())
    topic_genre_columns = [
        col for col in df.columns if col.startswith("topic") or col.startswith("genre")
    ]

    # This function takes two songs as input, represented as dictionaries
    # (song1 and song2). It extracts features from each song and computes the
    # Euclidean distance between their feature vectors. The feature vectors include
    # the 'age' feature of the songs and the features related to topics and genres,
    # which are stored in the topic_genre_columns.
    def custom_similarity(song1, song2):
        features_song1 = np.array(
            [song1["release_date"]] + [song1[col] for col in topic_genre_columns]
        )
        features_song2 = np.array(
            [song2["release_date"]] + [song2[col] for col in topic_genre_columns]
        )
        # euclidean dist b/w feature vectors
        distance = np.linalg.norm(features_song1 - features_song2)
        return distance

    def find_song_index(song_name):
        return df.index[df["song"] == song_name].tolist()

    input_song_name = input_song_name + " - " + input_artist_name
    input_song_index = find_song_index(input_song_name)

    if not input_song_index:
        print("Song not found.")
        exit()

    input_song_index = input_song_index[0]
    input_song_features = df.iloc[input_song_index]

    # Calling the custom_similairty and saving the data in a new frame so
    # we can sort it and recommend the top similar songs i.e ones with least distance
    similarity_scores = []

    for i in range(len(df)):
        song_name = df.iloc[i]["song"]
        similarity_score = custom_similarity(input_song_features, df.iloc[i])
        similarity_scores.append((song_name, similarity_score))

    # print("\n\nFirst 10 elements of similarity_scores:")
    # print(similarity_scores[:10])
    # print("\n\nLast 10 elements of similarity_scores:")
    # print(similarity_scores[-10:])

    sorted_similarity_scores = sorted(similarity_scores, key=lambda x: x[1])
    print("\n\nFirst 10 elements of sorted_similarity_scores:")
    print(sorted_similarity_scores[:10])
    print("\n\nLast 10 elements of sorted_similarity_scores:")
    print(sorted_similarity_scores[-10:])

    # Getting the song and artist name from the dataframe and appending to recommendations
    # frame which includes the final recommendations that are to be displayed to the user
    recommended_songs = [
        song_name for song_name, similarity_score in sorted_similarity_scores
    ]
    top_count = 20

    for index, song_name in enumerate(recommended_songs):
        recommendations.append({"song_name": song_name})
        print("song_name:", song_name)
        if index + 1 >= top_count:
            break
    recommendations.append({"song_name": "-------"})

    # First KNN layer based upon context
    from sklearn.neighbors import NearestNeighbors

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
    X_first_knn = df[first_knn_columns]
    first_knn = NearestNeighbors(n_neighbors=101)
    first_knn.fit(X_first_knn)

    input_song_neighbors_first = first_knn.kneighbors(
        [X_first_knn.iloc[input_song_index]], return_distance=False
    )[0]

    # Second KNN layer based upon the musical factors
    # The indices of the nearest neighbors to the input song are determined
    # by applying the second KNN model (second_knn) to the features of the input song.
    # The kneighbors method computes the nearest neighbors and returns their indices,
    # with return_distance=False indicating that only the indices of neighbors are required
    second_knn_columns = [
        "danceability",
        "loudness",
        "acousticness",
        "instrumentalness",
        "valence",
        "energy",
    ]

    X_second_knn = df.iloc[input_song_neighbors_first][second_knn_columns]
    second_knn = NearestNeighbors(n_neighbors=21)
    second_knn.fit(X_second_knn)

    input_song_neighbors_second = second_knn.kneighbors(
        [X_second_knn.iloc[0]], return_distance=False
    )[0]

    # retrieving indices of top 21 songs nearest to the input song
    top_20_song_indices = input_song_neighbors_second
    print("Names of the top 20 songs nearest to", input_song_name, ":")
    for i, song_index in enumerate(top_20_song_indices):
        song_name = df.iloc[song_index]["song"]
        recommendations.append({"song_name": song_name})
        print(i + 1, song_name)
    recommendations.append({"song_name": "-----"})

    # # Custom KNN
    # It computes the Euclidean distance between the query point and each
    # point in X, sorts the distances, and returns the indices of the k nearest neighbors.

    # Feature Extraction for First KNN: The columns for the first KNN are defined
    # (first_knn_columns), and features corresponding to these columns are extracted from
    # the dataset (df). These features are stored in X_first_knn.

    # Feature Extraction for Second KNN: Similarly, the columns for the second KNN are
    # defined (second_knn_columns), and features corresponding to these columns are extracted
    # from the dataset and stored in X_second_knn.
    import numpy as np

    def knn(X, query_point, k=5):
        distances = []
        for i in range(len(X)):
            distance = np.linalg.norm(X[i] - query_point)
            distances.append((i, distance))
        sorted_distances = sorted(distances, key=lambda x: x[1])
        nearest_neighbors_indices = [x[0] for x in sorted_distances[:k]]
        return nearest_neighbors_indices

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

    X_first_knn = df[first_knn_columns].values
    second_knn_columns = [
        "danceability",
        "loudness",
        "acousticness",
        "instrumentalness",
        "valence",
        "energy",
    ]

    X_second_knn = df[second_knn_columns].values

    # Nearest Neighbors Search:
    # Two instances of the custom KNN function are called to find the nearest neighbors.
    # First, the nearest neighbors to the input song are found using the first KNN model
    # (input_song_neighbors_first). Then, the query point for the second KNN model is set
    # as the features of the nearest neighbor found in the first KNN analysis.

    # This query point is used to find the nearest neighbors in the second KNN model (input_song_neighbors_second).
    def find_nearest_neighbors(X, query_point, k=5):
        distances = []
        for i in range(len(X)):
            distance = np.linalg.norm(X[i] - query_point)
            distances.append((i, distance))

        sorted_distances = sorted(distances, key=lambda x: x[1])
        nearest_neighbors_indices = [x[0] for x in sorted_distances[:k]]
        return nearest_neighbors_indices

    query_point_first_knn = X_first_knn[input_song_index]
    input_song_neighbors_first = knn(X_first_knn, query_point_first_knn, k=101)

    query_point_second_knn = X_second_knn[input_song_neighbors_first[0]]
    input_song_neighbors_second = knn(X_second_knn, query_point_second_knn, k=21)
    # print(df)

    # # Print Nearest Neighbor Names
    # the names of the top 20 songs nearest to the input song, as determined by the second
    # KNN analysis, are retrieved from the dataset and printed
    for i, song_index in enumerate(input_song_neighbors_second):
        song_name = df.iloc[song_index]["song"]
        recommendations.append({"song_name": song_name})
        print(i + 1, song_name)

    return jsonify({"recommendations": recommendations})


if __name__ == "__main__":
    app.run(debug=True)
