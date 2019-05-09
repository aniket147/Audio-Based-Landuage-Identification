from keras.models import load_model
from collections import Counter
import sys
import multiprocessing
import librosa
import numpy as np

SILENCE_THRESHOLD = .01
RATE = 24000
N_MFCC = 13
COL_SIZE = 30
N_CHRO = 13

def predict_class_audio(features, model):
    '''
    Predict class based on MFCC samples
    '''
    features = features.reshape(features.shape[0],features.shape[1],features.shape[2],1)
    y_predicted = model.predict_classes(features,verbose=0)
    return(Counter(list(y_predicted)).most_common(1)[0][0])

def predict_class_all(X_train, model):
    '''
    Prediction list
    '''
    predictions = []
    for mfcc in X_train:
        predictions.append(predict_class_audio(mfcc, model))
    return predictions

def to_mfcc(wav):
    '''
    Converts wav file to Mel Frequency Ceptral Coefficients
    '''
    return(librosa.feature.mfcc(y=wav, sr=RATE, n_mfcc=N_MFCC))

def to_chroma_cens(wav):
    '''
    Converts wav file to Chroma Energy Normalized
    '''
    return(librosa.feature.chroma_cens(y=wav, sr=RATE, n_chroma=N_CHRO))

def segment_one(mfcc):
    '''
    Creates segments from on mfcc image
    '''
    segments = []
    for start in range(0, int(mfcc.shape[1] / COL_SIZE)):
        segments.append(mfcc[:, start * COL_SIZE:(start + 1) * COL_SIZE])
    return(np.array(segments))

def create_segmented_mfccs(X_train):
    '''
    Creates segmented features from X_train
    '''
    segmented_mfccs = []
    for mfcc in X_train:
        segmented_mfccs.append(segment_one(mfcc))
    return(segmented_mfccs)

modelpath = sys.argv[1]

wavfilename = sys.argv[2]

classificationpath = modelpath[:-3] + '.txt'
f = open(classificationpath, 'r')
legend = f.read()
model = load_model(modelpath)

pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
y, sr = librosa.load(wavfilename)
X_test = librosa.core.resample(y=y,orig_sr=sr,target_sr=RATE, scale=True)

if sys.argv[3] == 'mfcc':
    X_test = pool.map(to_mfcc, [X_test])
else:
    X_test = pool.map(to_chroma_cens, [X_test])

y_predicted = predict_class_all(create_segmented_mfccs(X_test), model)
print(y_predicted)
print(legend)
