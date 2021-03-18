import numpy as np
import pandas as pd
from sklearn.utils import shuffle
import random
Random_Seed = 88
random.seed(Random_Seed)


def get_dataset_size(test_data_path, dataset, window_size):

    if dataset=='Air':
        data = pd.read_csv(test_data_path).values
        return  data.shape[0] - window_size
    elif dataset=='MIMIC-III':
        data = np.load(test_data_path,allow_pickle=True).tolist()['label']
        return len(data)
    elif dataset=='Boiler':
        data = pd.read_csv(test_data_path).values
        return data.shape[0] - window_size + 1


    else:
        raise Exception('unknown dataset!')


def data_transform(data_path, window_size, segments_length, dataset):
    '''
    transform data to the shape #[ samples_num, x_dim , segments_num , window_size, 1 ]
    :param data_path:
    :param window_size:
    :param segments_length:
    :param dataset:
    :return:
    '''

    if dataset == 'Boiler':
        data = pd.read_csv(data_path).values
        data = data[:, 2:]  #remove time step

        print('positive sample size:',sum(data[:,-1]))
        feature, label = [], []
        for i in range(window_size - 1, len(data)):
            label.append(data[i, -1])

            sample = []
            for length in segments_length:
                a = data[(i- length + 1):(i + 1), :-1]
                a = np.pad(a,pad_width=((0,window_size -length),(0,0)),mode='constant')# padding to [window_size, x_dim]
                sample.append(a)

            sample = np.array(sample)
            sample = np.transpose(sample,axes=((2,0,1)))[:,:,:,np.newaxis]

            feature.append(sample)

        feature, label = np.array(feature).astype(np.float32), np.array(label).astype(np.int32)

    elif dataset=='Air':
        data = pd.read_csv(data_path).values
        data = data[:, 1:]   #remove time step

        feature, label = [], []
        for i in range(window_size , len(data)):
            label.append(data[i, -1])
            sample = []
            for length in segments_length:
                a = data[(i - length):(i), :-1]
                a = np.pad(a, pad_width=((0, window_size - length), (0, 0)),
                           mode='constant')  # padding to [window_size, x_dim]
                sample.append(a)

            sample = np.array(sample)
            sample = np.transpose(sample, axes=((2, 0, 1)))[:, :, :,
                     np.newaxis]

            feature.append(sample)
        feature, label = np.array(feature).astype(np.float32), np.array(label).astype(np.float32)
    elif dataset=='MIMIC-III':
        data = np.load(data_path, allow_pickle=True).tolist()
        x, y = data['data'], data['label']
        print('positive sample size:', sum(y))

        feature, label = [], []
        for i in range(0, len(y)):
            label.append(y[i])
            sample = []
            for length in segments_length:
                a = x[i, -length:, :]
                a = np.pad(a, pad_width=((0, window_size - length), (0, 0)),
                           mode='constant')  # padding to [max_length, x_dim]
                sample.append(a)

            sample = np.array(sample)
            sample = np.transpose(sample, axes=((2, 0, 1)))[:, :, :,
                     np.newaxis]

            feature.append(sample)
        feature, label = np.array(feature).astype(np.float32), np.array(label).astype(np.int32)

    else:
        raise Exception('unknown dataset!')
    print(data_path, feature.shape)
    return feature, label


def data_generator(data_path, window_size, segments_length, batch_size, dataset, is_shuffle=False):

    print('data preparing..')

    feature, label = data_transform(data_path, window_size, segments_length, dataset)

    if is_shuffle:
        feature, label = shuffle(feature, label)

    batch_count = 0
    while True:
        if batch_size * batch_count >= len(label):
            batch_count = 0
            if is_shuffle:
                feature, label = shuffle(feature, label)

        start_index = batch_count * batch_size
        end_index = min(start_index + batch_size, len(label))
        batch_feature = feature[start_index: end_index]

        batch_label = label[start_index: end_index]
        batch_length = np.array(segments_length * (end_index - start_index))
        batch_count += 1

        yield batch_feature, batch_label, batch_length
