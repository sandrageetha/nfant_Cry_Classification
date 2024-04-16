import inspect

from flask import Flask, request, render_template, redirect, url_for
import pickle
import librosa
import numpy as np
# import scipy.fftpack
import os
from werkzeug.utils import secure_filename
from knn import KNN, euclidean_distance
from randomforest import RandomForest
from randomforest import SimpleDecisionTree

app = Flask(__name__, static_folder="static")
os.makedirs(os.path.join(app.root_path, 'static'), exist_ok=True)

'''@app.route('/login', methods =["GET", "POST"])

def login():
    global hupno
    if request.method == "POST":
       filename = request.form.get("hupfilename")
       audio_file_path = filename
       hupno=predict_audio_label(audio_file_path)
    return render_template('user.html')

@app.route('/results/<int:res>')
def results(hupno):
    return render_template('user.html',res=hupno)
from flask import Flask, request, render_template, redirect, url_for
# Assume predict_audio_label is defined somewhere
# from your_audio_processing_module import predict_audio_label

app = Flask(__name__)}'''


def predict_audio_label1(audio_path):
    # Load all the required models and scalers
    with open('knn_model.pkl', 'rb') as file:
        knn_model = pickle.load(file)
    with open('label_encoder.pkl', 'rb') as file:
        label_encoder = pickle.load(file)
    with open('min_max_scaler.pkl', 'rb') as file:
        scaler = pickle.load(file)

    # Extract features from the audio file
    features = extract_features(audio_path)

    if features is None:
        print("Feature extraction failed.")
        return None

    # Preprocess features (reshape and scale)
    features = features.reshape(1, -1)
    scaled_features = scaler.transform(features)

    # Predict the label using the trained KNN model
    try:
        predicted_label_num = knn_model.predict(scaled_features)[0]
        predicted_label = label_encoder.inverse_transform([predicted_label_num])
        print(f"Predicted label for the audio: {predicted_label[0]}")

        distances = [euclidean_distance(scaled_features, x) for x in knn_model.x_train]
        sorted_indices = np.argsort(distances)
        k_nearest_indices = sorted_indices[:knn_model.k]
        k_nearest_labels = knn_model.y_train[k_nearest_indices]
        confidence_score = np.mean(k_nearest_labels == predicted_label_num)
        print(f"Confidence score: {confidence_score}")

        return predicted_label[0], confidence_score * 100
    except Exception as e:
        print(f"Error during prediction: {e}")
        return None


def predict_audio_label2(audio_path):
    # Load all the required models and scalers
    with open('rf_model.pkl', 'rb') as file:
        model_rf = pickle.load(file)
    with open('label_encoder.pkl', 'rb') as file:
        label_encoder = pickle.load(file)
    with open('min_max_scaler.pkl', 'rb') as file:
        scaler = pickle.load(file)

    # Extract features from the audio file
    features = extract_features(audio_path)

    if features is None:
        print("Feature extraction failed.")
        return None

    # Preprocess features (reshape and scale)
    features = features.reshape(1, -1)
    scaled_features = scaler.transform(features)

    # Predict the label using the trained random forest model
    try:
        predicted_label_num = model_rf.predict(scaled_features)[0]
        predicted_label = label_encoder.inverse_transform([predicted_label_num])
        print(f"Predicted label for the audio: {predicted_label[0]}")

        # Calculate confidence score
        tree_predictions = [tree.predict(scaled_features) for tree in model_rf.trees]
        confidence_score = np.mean(tree_predictions == predicted_label_num)
        print(f"Confidence score: {confidence_score}")

        return predicted_label[0], confidence_score * 100
    except Exception as e:
        print(f"Error during prediction: {e}")
        return None


def predict_audio_label3(audio_path):
    # Load all the required models and scalers
    with open('svm_model.pkl', 'rb') as file:
        model_svm = pickle.load(file)
    with open('label_encoder.pkl', 'rb') as file:
        label_encoder = pickle.load(file)
    with open('min_max_scaler.pkl', 'rb') as file:
        scaler = pickle.load(file)

    # Extract features from the audio file
    features = extract_features(audio_path)

    if features is None:
        print("Feature extraction failed.")
        return None

    # Preprocess features (reshape and scale)
    features = features.reshape(1, -1)
    scaled_features = scaler.transform(features)

    # Predict the label using the trained SVM model
    try:
        predicted_label_num = model_svm.predict(scaled_features)[0]
        predicted_label = label_encoder.inverse_transform([predicted_label_num])
        print(f"Predicted label for the audio: {predicted_label[0]}")

        decision_values = model_svm.decision_function(scaled_features)
        # confidence_score = np.abs(decision_values[0])
        # confidence_score = np.max(np.abs(decision_values))
        # print(f"Confidence score: {confidence_score}")
        # return predicted_label[0],round(confidence_score*10,2)

        confidence_score = np.abs(decision_values[0, predicted_label_num])
        print(confidence_score)
        return predicted_label[0], round(confidence_score * 10, 2)
    except Exception as e:
        print(f"Error during prediction: {e}")
        return None


# Configuration for feature extraction
n_mfcc = 13
n_fft = 512  # Adjusted based on your frame duration needs
hop_length = 160  # Corresponds to a 10 ms frame shift at 16 kHz sampling rate
win_length = 512  # Typically set equal to n_fft
window = 'hann'
n_chroma = 12
n_mels = 128
n_bands = 7
fmin = 100


def extract_features(file_path, sr=22050):
    try:
        # Load the audio file
        y, sr = librosa.load(file_path, sr=sr)
        # Extract MFCC features with the updated n_mfcc value
        mfcc = np.mean(
            librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length, win_length=win_length,
                                 window=window).T, axis=0)
        # Extract Mel Spectrogram
        mel = np.mean(
            librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, win_length=win_length,
                                           window=window, n_mels=n_mels).T, axis=0)
        # Compute STFT
        stft = np.abs(librosa.stft(y))
        # Extract Chroma features
        chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sr).T, axis=0)
        # Extract Spectral Contrast
        contrast = np.mean(
            librosa.feature.spectral_contrast(S=stft, sr=sr, n_fft=n_fft, hop_length=hop_length, win_length=win_length,
                                              n_bands=n_bands, fmin=fmin).T, axis=0)
        # Extract Tonnetz features
        tonnetz = np.mean(librosa.feature.tonnetz(y=y, sr=sr).T, axis=0)
        # Combine all features into one vector
        return np.concatenate((mfcc, chroma, mel, contrast, tonnetz))
    except Exception as e:
        print(f"Error: Exception occurred in feature extraction for {file_path}: {e}")
        return None


@app.route('/')
def login():
    return render_template('h1.html')


app.config['UPLOAD_FOLDER'] = './static/files/'


@app.route('/login', methods=['POST'])
def submit():
    if request.method == 'POST':
        print(request.files)
        print(request.form)
        try:
            # Get the action type from the form
            button = request.form.get("btn_action")

            # Get the file from the form
            file = request.files["hupfilename"]
            if file:
                # Save the file
                filename = secure_filename(file.filename)
                audio_file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(audio_file_path)

                if button == "COMPARE":
                    # Predict using all models
                    knn_result, knn_score = predict_audio_label1(audio_file_path)
                    rf_result, rf_score = predict_audio_label2(audio_file_path)
                    svm_result, svm_score = predict_audio_label3(audio_file_path)

                    # Redirect to the comparison results page with all predictions
                    return render_template('user2.html', knn_result=knn_result, k_score=knn_score,
                                           rf_result=rf_result, rf_score=rf_score,
                                           svm_result=svm_result, sv_score=svm_score)

                else:
                    # Call the prediction function based on the action
                    if button == "KNN":
                        result, score = predict_audio_label1(audio_file_path)
                    elif button == "RF":
                        result, score = predict_audio_label2(audio_file_path)
                    elif button == "SVM":
                        result, score = predict_audio_label3(audio_file_path)

                    # Redirect to the results page with the prediction result
                    return redirect(url_for('results', res=result, sc=score))

        except Exception as e:
            print(e)
            return "Error processing the file", 500

    # If not POST request or any other issue, show the upload page again
    return render_template("secondpage.html")


@app.route('/results/<int:res>/<float:sc>')
def results(res, sc):
    return render_template('user1.html', result=res, score=sc)


@app.route('/secondpage')
def secondpage():
    return render_template('secondpage.html')


@app.route('/index')
def index():
    action = request.args.get('btn_action')
    return render_template('index.html', action=action)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
