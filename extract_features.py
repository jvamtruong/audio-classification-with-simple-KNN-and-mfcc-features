import os
import numpy as np
import json
import librosa

def load_data(data_path):
    with open(data_path, 'r') as fp:
        data = json.load(fp)
    X = np.array(data['mfcc'])
    y = np.array(data['labels'])
    return X, y

def extract_mfcc(file_path):
    signal, sr = librosa.load(file_path, sr=22050)
    mfcc = librosa.feature.mfcc(signal, sr=22050, n_mfcc=12, n_fft=2048, hop_length=512)
    mfcc = mfcc.T
    mfcc_pad = np.zeros((92, 12))
    mfcc_pad[:mfcc.shape[0], :] = mfcc
    return mfcc_pad

data = {
    'category': [],
    'mfcc': [],
    'labels': [],
    'path': []
}

json_path = 'json data.json'
dataset_path = 'raw data/'

if __name__ == '__main__':
    for index, (cur_path, dirs, files) in enumerate(os.walk(dataset_path)):
        if cur_path is not dataset_path:
            semantic_label = cur_path.split('/')[-1]
            data['category'].append(semantic_label)
            for file in files:
                file_path = os.path.join(cur_path, file)
                mfcc = extract_mfcc(file_path)
                data['mfcc'].append(mfcc.tolist())
                data['labels'].append(index - 1)
                data['path'].append(file_path)
                #print(f'ten file {file} | length signal {len(signal)} | duration {len(signal)/ sr} | mfcc shape {mfcc_pad.shape}')

    with open(json_path, "w") as fp:
        json.dump(data, fp, indent = 4)