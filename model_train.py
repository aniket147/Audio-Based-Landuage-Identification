import pandas as pd
from collections import Counter
import sys

from keras import utils
import librosa
import multiprocessing
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

from keras.callbacks import EarlyStopping, TensorBoard
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers.convolutional import MaxPooling2D, Conv2D
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator

SILENCE_THRESHOLD = .01
RATE = 24000
N_MFCC = 13
COL_SIZE = 30
EPOCHS = 10 #35#250
N_CHRO = 13

def split_people(df,test_size=0.2):
    '''
    Create train test split of DataFrame
    '''
    return train_test_split(df['language_num'],df['native_language'],test_size=test_size,random_state=1234)

def predict_class_audio(features, model):
    '''
    Predict class based on MFCC samples
    '''
    features = features.reshape(features.shape[0],features.shape[1],features.shape[2],1)
    y_predicted = model.predict_classes(features,verbose=0)
    return(Counter(list(y_predicted)).most_common(1)[0][0])


def predict_prob_class_audio(features, model):
    '''
    Predict class using MFCC 
    '''
    features = features.reshape(features.shape[0],features.shape[1],features.shape[2],1)
    y_predicted = model.predict_proba(features,verbose=0)
    return(np.argmax(np.sum(y_predicted,axis=0)))

def predict_class_all(X_train, model):
    '''
    Prediction list
    '''
    predictions = []
    for feature in X_train:
        predictions.append(predict_class_audio(feature, model))
    return predictions

def confusion_matrix(y_predicted,y_test):
    '''
    Create confusion matrix
    '''
    confusion_matrix = np.zeros((len(y_test[0]),len(y_test[0])),dtype=int )
    for index, predicted in enumerate(y_predicted):
        confusion_matrix[np.argmax(y_test[index])][predicted] += 1
    return(confusion_matrix)

def get_accuracy(y_predicted,y_test):
    '''
    Get accuracy
    '''
    c_matrix = confusion_matrix(y_predicted,y_test)
    return( np.sum(c_matrix.diagonal()) / float(np.sum(c_matrix)))

def to_chroma_cens(wav_file):
    '''
    Wav file to normalized chroma energy
    '''
    return(librosa.feature.chroma_cens(y=wav_file, n_chroma=N_CHRO, sr=RATE))

def to_mfcc(wav):
    '''
    Converts wav file to Mel Frequency Ceptral Coefficients
    '''
    return(librosa.feature.mfcc(y=wav, sr=RATE, n_mfcc=N_MFCC))

def to_categorical(y):
    '''
    One hot encoding
    '''
    lang_dict = {}
    for index,language in enumerate(set(y)):
        lang_dict[language] = index
    y = list(map(lambda x: lang_dict[x],y))
    return utils.to_categorical(y, len(lang_dict))

def get_wav(language_num):
    '''
    Load wav file
    '''
    y, sr = librosa.load('audio/{}.wav'.format(language_num))
    return(librosa.core.resample(y=y, orig_sr=sr, target_sr=RATE, scale=True))

def make_segments(features, labels):
    '''
    Segmenting of features and attach labels
    '''
    segments = []
    seg_labels = []
    for feature,label in zip(features,labels):
        for start in range(0, int(feature.shape[1] / COL_SIZE)):
            segments.append(feature[:, start * COL_SIZE:(start + 1) * COL_SIZE])
            seg_labels.append(label)
    return(segments, seg_labels)

def segment_one(feature):
    '''
    Segments from on feature image. 
    '''
    segments = []
    for start in range(0, int(feature.shape[1] / COL_SIZE)):
        segments.append(feature[:, start * COL_SIZE:(start + 1) * COL_SIZE])
    return(np.array(segments))

def create_segmented_features(X):
    '''
    Creates segmented features from X
    '''
    segmented_features = []
    for feature in X:
        segmented_features.append(segment_one(feature))
    return(segmented_features)


def train_model(X_train,y_train,X_valid,y_valid, batch_size=128): #64
    '''
    Train CNN
    '''

    rows = X_train[0].shape[0]
    cols = X_train[0].shape[1]
    valid_rows = X_valid[0].shape[0]
    val_columns = X_valid[0].shape[1]
    num_classes = len(y_train[0])

    input_shape = (rows, cols, 1)
    X_train = X_train.reshape(X_train.shape[0], rows, cols, 1 )
    X_valid = X_valid.reshape(X_valid.shape[0],valid_rows,val_columns,1)

    ksize = (3,3)
    model = Sequential()

    model.add(Conv2D(32, kernel_size=ksize, activation='relu',
                     data_format="channels_last",
                     input_shape=input_shape))

    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64,kernel_size=ksize, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer='adadelta',
                  metrics=['accuracy'])

    es = EarlyStopping(monitor='acc', min_delta=.005, patience=10, verbose=1, mode='auto')

    logpath = '../logs'
    tb = TensorBoard(log_dir=logpath, histogram_freq=0, batch_size=32, write_graph=True, write_grads=True,
                     write_images=True, embeddings_freq=0, embeddings_layer_names=None,
                     embeddings_metadata=None) # logging

    datagen = ImageDataGenerator(width_shift_range=0.05)

    model.fit_generator(datagen.flow(X_train, y_train, batch_size=batch_size),
                        steps_per_epoch=len(X_train) / 32
                        , epochs=EPOCHS,
                        callbacks=[es,tb], validation_data=(X_valid,y_valid))
    return (model)

def saving(model, model_filename, mapping):
    '''
    Save model to file
    '''
    f = open('models/{}.txt'.format(model_filename), 'w+')
    f.write(str(mapping))
    f.close()
    model.save('models/{}.h5'.format(model_filename))  # creates a HDF5 file 'my_model.h5'


if __name__ == '__main__':
    file_name = sys.argv[1]
    model_filename = sys.argv[2]
    df = pd.read_csv(file_name)

    X_train, X_test, y_train, y_test = split_people(df)

    train_count = Counter(y_train)
    test_count = Counter(y_test)

    a =  y_train
    # To categorical
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)
    mapping = {}
    for i,x in enumerate(list(a)):
        mapping[x] = np.argmax(y_train[i])

    # Not necessarily required pool
    print('Loading wav Files')
    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
    X_train = pool.map(get_wav, X_train)
    X_test = pool.map(get_wav, X_test)
    print("Done Loading wav files...")

    print('Extracting Feature....')
    X_train = pool.map(to_chroma_cens, X_train)
    X_test = pool.map(to_chroma_cens, X_test)
    # X_train = pool.map(to_mfcc, X_train)
    # X_test = pool.map(to_mfcc, X_test)

    X_train, y_train = make_segments(X_train, y_train)
    X_valid, y_valid = make_segments(X_test, y_test)

    X_train, _, y_train, _ = train_test_split(X_train, y_train, test_size=0)

    model = train_model(np.array(X_train), np.array(y_train), np.array(X_valid), np.array(y_valid))

    y_pred = predict_class_all(create_segmented_features(X_test), model) # Predictions

    print('Confusion matrix of total samples:\n', np.sum(confusion_matrix(y_pred, y_test),axis=1))
    print('Confusion matrix:\n',confusion_matrix(y_pred, y_test))
    print('Accuracy:', get_accuracy(y_pred, y_test))

    saving(model, model_filename, mapping)
