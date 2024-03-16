import requests
import base64
import random
import os
from dotenv import load_dotenv
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Spotify API credentials
# include genre in cosine similarity
# weights to some features
load_dotenv()
CLIENT_ID = os.getenv("CLIENT_ID")
CLIENT_SECRET = os.getenv("CLIENT_SECRET")

client_credentials_manager = SpotifyClientCredentials(
    client_id=CLIENT_ID,
    client_secret=CLIENT_SECRET,
)
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

# Define three hashmaps
similarity_hashmap1 = {}
similarity_hashmap2 = {}
similarity_hashmap3 = {}


def search_track(song_name, artist):
    url = "https://api.spotify.com/v1/search"
    headers = {
        "Authorization": f"Bearer {get_access_token()}",
    }
    params = {"q": f"track:{song_name} artist:{artist}", "type": "track", "limit": 1}

    response = requests.get(url, headers=headers, params=params)

    if response.status_code == 200:
        data = response.json()
        if data["tracks"]["items"]:
            return data["tracks"]["items"]
        else:
            print("Track not found.")
            return None
    else:
        print(f"Error: {response.status_code}")
        print(f"Response: {response.text}")
        return None


def calculate_cosine_similarity(track1_features, track2_features):
    # Extract relevant numerical features for similarity calculation
    features1 = np.array(
        [
            track1_features["energy"],
            track1_features["key"],
            track1_features["loudness"],
            track1_features["mode"],
            track1_features["speechiness"],
            track1_features["acousticness"],
            track1_features["valence"],
            track1_features["tempo"],
        ]
    )
    features2 = np.array(
        [
            track2_features["energy"],
            track2_features["key"],
            track2_features["loudness"],
            track2_features["mode"],
            track2_features["speechiness"],
            track2_features["acousticness"],
            track2_features["valence"],
            track2_features["tempo"],
        ]
    )

    # Reshape features for cosine similarity calculation
    features1 = features1.reshape(1, -1)
    features2 = features2.reshape(1, -1)

    # Calculate cosine similarity
    similarity = cosine_similarity(features1, features2)[0][0]
    return similarity


def get_audio_features(track_id):
    track_features = sp.audio_features(tracks=[track_id])[0]
    return track_features


def display_track_details(track):
    if not track:
        return None

    track = track[0]  # Take the first search result

    # Get track ID
    track_id = track["id"]

    # Fetch track features
    track_features = get_track_features(track_id)
    return track_features


def get_track_features(track_id):
    url = f"https://api.spotify.com/v1/audio-features/{track_id}"
    headers = {
        "Authorization": f"Bearer {get_access_token()}",
    }

    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        data = response.json()
        return data
    else:
        print(f"Error: {response.status_code}")
        print(f"Response: {response.text}")
        return None


def get_access_token():
    client_id = os.getenv("CLIENT_ID")
    client_secret = os.getenv("CLIENT_SECRET")

    auth_string = f"{client_id}:{client_secret}"
    auth_bytes = auth_string.encode("utf-8")
    auth_base64 = base64.b64encode(auth_bytes).decode("utf-8")

    url = "https://accounts.spotify.com/api/token"
    headers = {
        "Authorization": f"Basic {auth_base64}",
        "Content-Type": "application/x-www-form-urlencoded",
    }
    data = {"grant_type": "client_credentials"}

    response = requests.post(url, headers=headers, data=data)

    if response.status_code == 200:
        json_result = response.json()
        if "access_token" in json_result:
            return json_result["access_token"]
        else:
            print("Access token not found in response")
    else:
        print(f"Error: {response.status_code}")
        print(f"Response: {response.text}")

    return None


def get_random_tracks(access_token):
    url = "https://api.spotify.com/v1/browse/featured-playlists"
    headers = {"Authorization": f"Bearer {access_token}"}
    # Fetch featured playlists
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        playlists = response.json().get("playlists", {}).get("items", [])
        if playlists:
            # Select 5 random playlists
            selected_playlists = random.sample(playlists, 20)
            random_tracks = []
            for playlist in selected_playlists:
                # Fetch tracks from each selected playlist
                playlist_id = playlist["id"]
                playlist_name = playlist["name"]
                playlist_url = (
                    f"https://api.spotify.com/v1/playlists/{playlist_id}/tracks"
                )
                response = requests.get(playlist_url, headers=headers)
                if response.status_code == 200:
                    tracks = response.json().get("items", [])
                    if tracks:
                        # Randomly select 4 tracks from each playlist
                        random_tracks.extend(random.sample(tracks, 2))
                    else:
                        print(f"No tracks found in playlist: {playlist_name}")
                else:
                    print(f"Error fetching playlist tracks: {response.status_code}")
            return random_tracks
        else:
            print("No playlists found.")
    else:
        print(f"Error fetching featured playlists: {response.status_code}")
    return None


def sort_hashmap_by_values(hashmap):
    sorted_hashmap = {
        k: v for k, v in sorted(hashmap.items(), key=lambda item: item[1], reverse=True)
    }
    return sorted_hashmap


def display_top_10_songs(hashmap):
    sorted_hashmap = sort_hashmap_by_values(hashmap)
    top_10_songs = dict(list(sorted_hashmap.items())[:10])
    print("Top 10 Songs:")
    for song, similarity in top_10_songs.items():
        print(f"{song}: {similarity}")


def merge_and_sort_hashmaps(hashmap1, hashmap2, hashmap3):
    combined_hashmap = {**hashmap1, **hashmap2, **hashmap3}
    sorted_combined_hashmap = dict(
        sorted(combined_hashmap.items(), key=lambda item: item[1], reverse=True)
    )
    return sorted_combined_hashmap


def calculate_and_return_hashmap(input_track_features, access_token):
    similarity_hashmap = {}
    if access_token:
        random_tracks = get_random_tracks(access_token)
        if random_tracks:
            for i, random_track in enumerate(random_tracks, 1):
                random_track_info = random_track.get("track", {})
                random_track_name = random_track_info.get("name")
                random_track_features = get_track_features(random_track_info["id"])
                similarity = calculate_cosine_similarity(
                    input_track_features, random_track_features
                )
                similarity_hashmap[random_track_name] = similarity
                print(f"Track {i}: {random_track_name}, Similarity: {similarity}")
    return similarity_hashmap


# Main function
def main():
    song_name = input("Enter the name of the song: ")
    artist = input("Enter the name of the artist: ")

    # Search for the track
    track = search_track(song_name, artist)
    input_track_features = display_track_details(track)

    access_token = get_access_token()
    if access_token:
        similarity_hashmap1 = calculate_and_return_hashmap(
            input_track_features, access_token
        )
        similarity_hashmap2 = calculate_and_return_hashmap(
            input_track_features, access_token
        )
        similarity_hashmap3 = calculate_and_return_hashmap(
            input_track_features, access_token
        )

        # print("Top 10 Songs from Hashmap 1:")
        # display_top_10_songs(similarity_hashmap1)
        # print("\nTop 10 Songs from Hashmap 2:")
        # display_top_10_songs(similarity_hashmap2)
        # print("\nTop 10 Songs from Hashmap 3:")
        # display_top_10_songs(similarity_hashmap3)

        sorted_combined_hashmap = merge_and_sort_hashmaps(
            similarity_hashmap1, similarity_hashmap2, similarity_hashmap3
        )

        print("\nTop 10 Songs from Combined and Sorted Hashmaps:")
        display_top_10_songs(sorted_combined_hashmap)


if __name__ == "__main__":
    main()
