import os
import numpy as np
import json
import librosa

data = {
    'category': [],
    'mfcc': [],
    'labels': []
}

json_path = 'json data.json'
dataset_path = 'raw data/'

for index, (cur_path, dirs, files) in enumerate(os.walk(dataset_path)):
    if cur_path is not dataset_path:
        semantic_label = cur_path.split('/')[-1]
        data['category'].append(semantic_label)
        for file in files:
            file_path = os.path.join(cur_path, file)
            signal, sr = librosa.load(file_path, sr = 22050)
            mfcc = librosa.feature.mfcc(signal, sr = 22050, n_mfcc = 12, n_fft = 2048, hop_length = 512)
            mfcc = mfcc.T
            mfcc_pad = np.zeros((92, 12))
            mfcc_pad[:mfcc.shape[0], :] = mfcc[:, :]
            data['mfcc'].append(mfcc_pad.tolist())
            data['labels'].append(index - 1)
            #print(f'ten file {file} | length signal {len(signal)} | duration {len(signal)/ sr} | mfcc shape {mfcc_pad.shape}')

with open(json_path, "w") as fp:
    json.dump(data, fp, indent = 4)
