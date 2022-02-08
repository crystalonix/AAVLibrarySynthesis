import numpy as np
import pickle

positive_data_file = 'D_Assembled_20-07-28.txt'
negative_date_file = 'D_Unassembled_20-07-28.txt'
SEQUENCE_LENGTH = 33
NUMBER_OF_ASSEMBLED_SEQUENCES = 14668726
NUMBER_OF_UNASSEMBLED_SEQUENCES = 7175879
ONE_HOT_DIMENSION = 26
WILD_TYPE_SEQUENCE = 'TPSTTQRQKTNNNTKKEEKQGSEKTNVDIEKRR'
neg_sequences = None
pos_sequences = None


def load_entire_data(input_file, start_index=0, end_index=-1, sequence_len=SEQUENCE_LENGTH):
    with open(input_file) as fl:
        sequences = fl.readlines()
    print(f'length of the file {len(sequences)}')
    sequence_data = np.array([sequence_to_number(sequences[start_index], sequence_len)])
    if end_index == -1:
        end_index = len(sequences)
    for i in range(start_index + 1, min(end_index, len(sequences))):
        if i % 10000 == 0:
            print(f'sequence at itr:{i} is {sequences[i]}')
        sequence_data = np.r_[sequence_data, sequence_to_number(sequences[i], sequence_len).reshape(1, -1)]
    return sequence_data


def load_random_data_samples(number_of_samples, randomize=True, seed=-1):
    """
    This method returns a random subsample of positive and negative data points;
    This method returns en equal proportion of positive and negative random samples
    :param seed:
    :param number_of_samples:
    :type randomize: object
    :rtype: object
    """
    # first load number_of_samples/2 positive samples
    if seed >= 0:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()
    pi1 = rng.permutation(NUMBER_OF_ASSEMBLED_SEQUENCES)[0:int(number_of_samples / 2)]
    pi2 = rng.permutation(NUMBER_OF_UNASSEMBLED_SEQUENCES)[0:int(number_of_samples / 2)]
    with open(positive_data_file) as fl:
        sequences = fl.readlines()
    sequence_data = np.array([sequence_to_number(sequences[pi1[0]], SEQUENCE_LENGTH)])
    for k in range(1, len(pi1)):
        sequence_data = np.r_[sequence_data, sequence_to_number(sequences[pi1[k]], SEQUENCE_LENGTH).reshape(1, -1)]

    with open(negative_date_file) as fl:
        sequences = fl.readlines()
    for k in range(len(pi2)):
        sequence_data = np.r_[sequence_data, sequence_to_number(sequences[pi2[k]], SEQUENCE_LENGTH).reshape(1, -1)]

    labels = np.ones(len(pi1))
    labels = np.r_[labels, np.zeros(len(pi2))]
    sequence_data = np.c_[sequence_data, labels]
    if randomize:
        sequence_data = randomly_shuffle_data(sequence_data)
    return sequence_data


def load_random_data_samples_in_batches(batch_id, number_of_batches, batch_size, randomize=True, seed=-1):
    """
    This method returns a random subsample of positive and negative data points;
    This method returns en equal proportion of positive and negative random samples
    :param seed:
    :param number_of_samples:
    :type randomize: object
    :rtype: object
    """
    # first load number_of_samples/2 positive samples
    number_of_samples = number_of_batches * batch_size
    if seed >= 0:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()
    p1 = rng.permutation(NUMBER_OF_ASSEMBLED_SEQUENCES)[0:int(number_of_samples / 2)]
    p2 = rng.permutation(NUMBER_OF_UNASSEMBLED_SEQUENCES)[0:int(number_of_samples / 2)]

    pi_batch_pos = p1[int(batch_id * (batch_size / 2)):int((batch_id + 1) * batch_size / 2)]
    pi_batch_neg = p2[int(batch_id * batch_size / 2):int((batch_id + 1) * batch_size / 2)]
    global pos_sequences, neg_sequences
    if pos_sequences is None:
        with open(positive_data_file) as fl:
            pos_sequences = fl.readlines()
    sequence_data = np.array([sequence_to_number(pos_sequences[pi_batch_pos[0]], SEQUENCE_LENGTH)])
    for k in range(1, len(pi_batch_pos)):
        sequence_data = np.r_[
            sequence_data, sequence_to_number(pos_sequences[pi_batch_pos[k]], SEQUENCE_LENGTH).reshape(1, -1)]

    if neg_sequences is None:
        with open(negative_date_file) as fl:
            neg_sequences = fl.readlines()
    # while True:
    #     pass
    for k in range(len(pi_batch_neg)):
        sequence_data = np.r_[
            sequence_data, sequence_to_number(neg_sequences[pi_batch_neg[k]], SEQUENCE_LENGTH).reshape(1, -1)]

    labels = np.ones(len(pi_batch_pos))
    labels = np.r_[labels, np.zeros(len(pi_batch_neg))]
    sequence_data = np.c_[sequence_data, labels]
    if randomize:
        sequence_data = randomly_shuffle_data(sequence_data)
    print('came here')
    while True:
        pass
    return sequence_data


def load_random_data_samples_with_mutation_dist(number_of_samples, randomize=True, seed=-1):
    """
    This method returns a random subsample of positive and negative data points;
    This method returns en equal proportion of positive and negative random samples
    :param seed:
    :param number_of_samples:
    :type randomize: object
    :rtype: object
    """
    random_data_points = load_random_data_samples(number_of_samples, randomize, seed)
    distances = np.zeros(len(random_data_points))
    for i, x in enumerate(random_data_points):
        mutation_dist = get_mutation_distance(x[:SEQUENCE_LENGTH])
        distances[i] = mutation_dist
    return random_data_points, distances


def convert_dist_to_dictionary(distances):
    dist_dict = {}
    for i in distances:
        if i in dist_dict:
            dist_dict[i] = dist_dict[i] + 1
        else:
            dist_dict[i] = 1
    return dist_dict


def filter_data_points_on_mutation_distance(data_points, distances, max_dist, min_dist=0):
    """
    Filters the data points on distances
    :param data_points:
    :param distances:
    :param max_dist:
    :param min_dist:
    :return:
    """
    filtered_data_set = data_points[np.where(distances <= max_dist)]
    filtered_distances = distances[np.where(distances <= max_dist)]
    return filtered_data_set[np.where(filtered_distances >= min_dist)], filtered_distances
    [np.where(filtered_distances >= min_dist)]


def randomly_shuffle_data(sequence_data):
    """
    This method will randomly shuffle all the data points
    :param sequence_data:
    :return:
    """
    ind = np.arange(sequence_data.shape[0])
    np.random.shuffle(ind)
    return sequence_data[ind, :]


def get_mutation_distance(numerical_sequence):
    wild_type_vector = sequence_to_number(WILD_TYPE_SEQUENCE, SEQUENCE_LENGTH)
    dist = (wild_type_vector == numerical_sequence)
    return np.sum(dist)


def sequence_to_number(sequence, sequence_length):
    numerical_representation = np.zeros(sequence_length)
    for i in range(sequence_length):
        numerical_representation[i] = ord(sequence[i]) - ord('A')
    return numerical_representation


def write_numerical_sequences(input_file, file_name, start_index, end_index, frmt='%2.0f'):
    num_sequences = load_entire_data(input_file, start_index, end_index)
    np.savetxt(file_name, num_sequences, fmt=frmt)


def split_data_into_train_val_test(data, proportions, shuffle=False, dists=None):
    """
    This method splits the given dataset appropriately into train, validation and test sets
    :param dists: mutation distances
    :return:
    :param proportions:
    :return:
    :param data:
    :param shuffle:
    """
    if shuffle:
        pass
    else:
        data_len = data.shape[0]
        tr_len = int(proportions[0] * data_len / np.sum(proportions))
        val_len = int(proportions[1] * data_len / np.sum(proportions))
        test_len = data_len - tr_len - val_len

        train_x = data[0:tr_len, :-1]
        train_y = data[0:tr_len, -1]
        validation_x = data[tr_len:tr_len + val_len, :-1]
        validation_y = data[tr_len:tr_len + val_len, -1]
        test_x = data[tr_len + val_len:tr_len + val_len + test_len, :-1]
        test_y = data[tr_len + val_len:tr_len + val_len + test_len, -1]

        dist_tr = None
        dists_val = None
        dists_tst = None
        if dists is not None:
            dist_tr = dists[0:tr_len]
            dists_val = dists[tr_len:tr_len + val_len]
            dists_tst = dists[tr_len + val_len:tr_len + val_len + test_len]
        return train_x, train_y, validation_x, validation_y, test_x, test_y, dist_tr, dists_val, dists_tst


def one_hot_encode(x, features_to_encode=None):
    if features_to_encode is None:
        features_to_encode = np.ones(x.shape[1])
        features_to_encode[-1] = 0
    z = None
    for i, encode in enumerate(features_to_encode):
        xi = x[:, i]
        if encode:
            xiint = xi.astype(int)
            nv = ONE_HOT_DIMENSION  # np.max(xiint) + 1
            oh = np.eye(nv)[xiint]
        else:
            oh = xi.copy().reshape(-1, 1)
        if z is None:
            z = oh
        else:
            z = np.concatenate((z, oh), axis=1)
    return z


def one_hot_encode_column(col, one_hot_dim=None):
    """
    Returns the one hot representation of a col
    :param col:
    :param one_hot_dim: dim of one hot representation, if not provided as argument it will pick the max
    :return:
    """
    if one_hot_dim is None:
        one_hot_dim = np.max(col) - np.min(col) + 1
    col_int = col.astype(int)
    oh = np.eye(one_hot_dim)[col_int - np.min(col_int)]
    return oh


def save_model_in_pickle(file_name, model):
    with open(file_name, 'wb') as file:
        pickle.dump(model, file)

# file_ind = 2
# data_vol = 30000
# # for file_ind in range(5):
# data_dict = eval(open('data-dist' + str(file_ind)+'-'+str(data_vol)+'.txt', 'r').read())
# accr_dict = eval(open('accuracy-dist' + str(file_ind)+'-'+str(data_vol)+'.txt', 'r').read())
# plot_utils.plot_linechart(data_dict, 'data-dist' + str(file_ind)+'-'+str(data_vol)+'.png')
# plot_utils.plot_linechart(accr_dict, 'accuracy-dist' + str(file_ind)+'-'+str(data_vol)+'.png')
# data = load_random_data_samples(6, False, 0)
# np.savetxt('subsample_sequences.txt', data, fmt='%2.0f')
# data = load_random_data_samples(2, True, 0)
# data = one_hot_encode(data)
# print(data[0])
# np.savetxt('random_sequences.txt', data, fmt='%2.0f')
# split_len = 100000
# assembled_itr = int(NUMBER_OF_ASSEMBLED_SEQUENCES / split_len) + 1
# unassembled_itr = int(NUMBER_OF_UNASSEMBLED_SEQUENCES / split_len) + 1
# print(f'{assembled_itr}, {unassembled_itr}')
# for i in range(assembled_itr):
#     write_numerical_sequences(positive_data_file, 'assembled_numerical_sequences_' + str(i) + '.txt', i * split_len,
#                               (i + 1) * split_len)
#     print(f'step {i} has been written out')
#
# for i in range(unassembled_itr):
#     write_numerical_sequences(negative_date_file, 'unassembled_numerical_sequences_' + str(i) + '.txt', i * split_len,
#                               (i + 1) * split_len)
#     print(f'step {i} has been written out')
# sample_batch = load_random_data_samples_in_batches(1, number_of_batches=100, batch_size=1000, randomize=True, seed=-1)
# print(f'size of batch: {len(sample_batch)}')
