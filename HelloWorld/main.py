import pickle
from flask import Flask, request, jsonify
# from tensorflow import keras
import spotipy
import sklearn
from spotipy import SpotifyClientCredentials, util
import numpy as np

client_id = '30119fb867344911a90685424c5aac32'
client_secret = '71abed44a5584ac6a4945d829be66964'
scope = 'playlist-modify-public'

manager = SpotifyClientCredentials(client_id, client_secret)
sp = spotipy.Spotify(client_credentials_manager=manager)

# model = keras.models.load_model("model.h5")
model = pickle.load(open('model.pkl','rb'))

app = Flask(__name__)

def get_songs_features(ids):
    meta = sp.track(ids)
    features = sp.audio_features(ids)

    # meta
    name = meta['name']
    album = meta['album']['name']
    artist = meta['album']['artists'][0]['name']
    release_date = meta['album']['release_date']
    length = meta['duration_ms']
    popularity = meta['popularity']
    ids = meta['id']

    # features
    acousticness = features[0]['acousticness']
    danceability = features[0]['danceability']
    energy = features[0]['energy']
    instrumentalness = features[0]['instrumentalness']
    liveness = features[0]['liveness']
    valence = features[0]['valence']
    loudness = features[0]['loudness']
    speechiness = features[0]['speechiness']
    tempo = features[0]['tempo']
    key = features[0]['key']
    time_signature = features[0]['time_signature']

    track = [name, album, artist, ids, release_date, popularity, length, danceability, acousticness,
             energy, instrumentalness, liveness, valence, loudness, speechiness, tempo, key, time_signature]
    columns = ['name', 'album', 'artist', 'id', 'release_date', 'popularity', 'length', 'danceability', 'acousticness',
               'energy', 'instrumentalness',
               'liveness', 'valence', 'loudness', 'speechiness', 'tempo', 'key', 'time_signature']
    return track, columns

def predict_mood(id_song):

    preds = get_songs_features(id_song)
    preds_features = np.array(preds[0][6:-2]).reshape(-1,1).T
    results = model.predict(preds_features)
    return results

@app.route('/')
def home():
    return "Hello World"


@app.route('/predict', methods=['POST'])
def predict():
    mood = ["Calm","Energetic","Happy","Sad"]
    song = request.form.get('song')
    abc = predict_mood(song)
    print(abc[0])
    print(mood[abc[0]])
    # for i in range(0,4):
    #     if abc[0][i] == 1:
    #         return jsonify(mood[i])
    return jsonify({'mood':mood[abc[0]]})


if __name__ == '__main__':
    app.run(debug=True)
