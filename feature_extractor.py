import pandas as pd
import numpy as np
import common_utils

features_map = [
                # 'Feature#1 Index',
                'Feature#2 Carbon Count'
                ]
SEQUENCE_SHORT_HAND_HEADER = 'Short Hand'
is_categorical = [
                    # True,
                    False
                  ]
sequence_index_lookup = {}


def read_features_from_excel():
    all_excel_features = pd.read_excel('protein_features.xlsx')
    for i in range(len(all_excel_features[features_map[0]])):
        sequence_index_lookup[all_excel_features[SEQUENCE_SHORT_HAND_HEADER][i]] = i
    print(f'check the look up map: {sequence_index_lookup}')
    features_array = None
    for ind, feature in enumerate(features_map):
        this_feature_values = all_excel_features[feature].to_numpy()
        if is_categorical[ind]:
            this_feature_values = common_utils.one_hot_encode_column(this_feature_values)
        else:
            this_feature_values = this_feature_values.reshape(-1, 1)
        if features_array is None:
            features_array = this_feature_values
        else:
            features_array = np.concatenate((features_array, this_feature_values), axis=1)
    # print(f'check the final features array:\n {features_array}')
    return features_array


def sequence_to_features_array(sequence, features_array):
    index = sequence_index_lookup[sequence]
    return features_array[index]


def extract_features_for_sequences(data_seq, features_array):
    all_seq_reps = None
    for row in range(len(data_seq)):
        z = None
        for c in data_seq[row]:
            this_seq_rep = sequence_to_features_array(c, features_array)
            if z is None:
                z = this_seq_rep
            else:
                z = np.concatenate((z, this_seq_rep))
        if all_seq_reps is None:
            all_seq_reps = z.reshape(1, -1)
        else:
            all_seq_reps = np.concatenate((all_seq_reps, z.reshape(1, -1)))
    return all_seq_reps


fts_array = read_features_from_excel()
data_sq = np.array([['A', 'Q'], ['C', 'D']])
print(extract_features_for_sequences(data_sq, fts_array))
