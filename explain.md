## start, mostly testing hard coded

songs.csv taken from kagggle with around 28,000 songs

testing, 20 something songs using spotify api

## data preprocessing

saved in track_features.json ( )audio info) and track_info.json ()album, genre etc) and then mergering later.

## Feature Engineering

after minmax, onehot encoding for **topic, genre, audio.**  we do content based filtering on the result of this encoding.

---

## we have calcualted similarity, NOT using built in func

def custom_similarity(song1, song2):
    # Extract features from songs
    features_song1 = np.array(
        [song1["age"]] + [song1[col] for col in topic_genre_columns]
    )
    features_song2 = np.array(
        [song2["age"]] + [song2[col] for col in topic_genre_columns]
    )

    # Calculate Euclidean distance between the feature vectors
    distance = np.linalg.norm(features_song1 - features_song2)

    return distance

* **Purpose** : This function calculates the similarity between two songs based on their features.
* **Parameters** :
* `song1`: The first song (as a dictionary or similar data structure containing its features).
* `song2`: The second song (similarly structured dictionary).
* **Steps** :

1. Extract the features from the songs. It appears to include the "age" feature and features related to topics and genres.
2. Convert the features into NumPy arrays (`features_song1` and `features_song2`).
3. Calculate the Euclidean distance between the feature vectors, which represents the similarity between the songs.
4. Return the calculated distance.

### Retrieving Column Names

//Get the column names for 'topic' and 'genre' after one-hot encoding
topic_genre_columns = [
    col for col in df.columns if col.startswith("topic") or col.startswith("genre")
]

* **Purpose** : This code snippet retrieves the column names related to topics and genres from the DataFrame `df`.
* **Steps** :

1. It iterates over the columns of the DataFrame `df`.
2. It selects columns that start with either "topic" or "genre" and stores their names in the list `topic_genre_columns`.

In summary, the `custom_similarity` function computes the similarity between two songs based on their features, including age, topics, and genres. The column names related to topics and genres are extracted from the DataFrame `df` and used within this function. This custom similarity function can be utilized in various tasks such as recommendation systems or clustering.

## 1st knn algo which recommneds first 20 songs is built in.

LOGIC:

Imagine you have a massive music library, and you're trying to find songs similar to a particular song you love. However, you're not entirely sure what makes a song similar to another. So, you decide to use two layers of friends (KNN models) to help you.

### First KNN Layer (Prioritizing songs based on context)

1. **Data Points** : Each friend in this layer represents a song in your library. Each friend knows some characteristics of the songs, like whether they're romantic, sad, or energetic.
2. **Features** : These characteristics of the songs are like the topics you might discuss with your friends about a song.
3. **Decision Making** : When you ask a question about a song, your friends (the KNN model) look at the characteristics of that song and compare them to other songs they know. They then find the songs that are most similar to the one you asked about based on those characteristics.
4. **k Value** : You decide to ask your 100 nearest friends about the song because you want a broad range of opinions.

### Second KNN Layer (Prioritizing songs based on musical factors)

1. **Data Points** : Now, you have a shorter list of songs recommended by your first set of friends.
2. **Features** : These songs are characterized by their musical features like danceability, loudness, etc.
3. **Decision Making** : Your second set of friends (the second KNN model) look at the musical features of the songs recommended by the first set of friends. They then find the songs that are most similar to your original song based on those musical features.
4. **k Value** : You decide to ask your 20 nearest friends about these songs because you want a more focused set of opinions.

### Recommended Songs

After going through both layers, you get a list of the top 20 songs recommended by your second set of friends. These songs are the ones that both match the context and have similar musical characteristics to your original song.

So, by using two layers of friends (KNN models), you're able to find songs in your library that are both thematically and musically similar to the one you love.

---

## 2nd knn is written by us:

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

* **Purpose** : This function calculates the k nearest neighbors of a query point within a dataset `X`.
* **Parameters** :
* `X`: The dataset containing the features of the data points.
* `query_point`: The feature vector of the query point for which we want to find the nearest neighbors.
* `k`: The number of nearest neighbors to find (default is 5).
* **Steps** :

1. Iterate through each data point in `X`.
2. Calculate the Euclidean distance between the query point and each data point.
3. Store the distances along with their indices in a list.
4. Sort the distances in ascending order.
5. Select the indices of the first `k` data points with the smallest distances.
6. Return the indices of the nearest neighbors.

### Finding Nearest Neighbors for Music Recommendation

* **Data Preparation** :
* Two sets of features are defined for the kNN models: `first_knn_columns` and `second_knn_columns`.
* Features are extracted from the DataFrame `df` and converted to NumPy arrays (`X_first_knn` and `X_second_knn`).
* **Finding Neighbors** :
* For the first KNN:
  * The features of the input song (`query_point_first_knn`) are used to find the 100 nearest neighbors (`input_song_neighbors_first`).
* For the second KNN:
  * The features of the nearest neighbors found by the first KNN are used as query points to find the 20 nearest neighbors (`input_song_neighbors_second`).

### Output

* The recommendations are printed with the song names retrieved from the DataFrame `df`.

In essence, this code uses the kNN algorithm to find similar songs based on their features and recommends them to the user. It's a manual implementation of kNN, allowing for more control and customization compared to using libraries like scikit-learn.
