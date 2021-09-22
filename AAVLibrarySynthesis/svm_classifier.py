import numpy as np
from sklearn.svm import LinearSVC, SVC
from aavgit.AAVLibrarySynthesis import common_utils


def model_accuracy(model, x, true_y):
    """
    This method returns the accuracy of the model
    :param model:
    :param x:
    :param true_y:
    :return:
    """
    pred = model.predict(x)
    return np.sum(pred == true_y) / true_y.shape[0]


def majority_voting_model_accuracy(models, x, true_y):
    """
    This method evaluates the performance of the meta model takes the majority voting from svm models
    :param models:
    :param x:
    :param true_y:
    :return:
    """
    z = None
    for k in range(len(models)):
        predictions = models[k].predict(x)
        if z is None:
            z = predictions
        else:
            z = np.c_[z, predictions]
    majority_votes = np.sum(z, axis=1)
    final_votes = majority_votes >= len(models) / 2
    return np.sum(final_votes == true_y) / true_y.shape[0]


def evaluate_model(name, model, train_x, train_y, validation_x, validation_y):
    """
    Evaluates the model based on training and validation accuracy
    :param validation_y:
    :param name:
    :param model:
    :param train_x:
    :param train_y:
    :param validation_x:
    :return:
    """
    train_acc = model_accuracy(model, train_x, train_y)
    val_acc = model_accuracy(model, validation_x, validation_y)
    print('{}: \n\t--- Training accuracy: {:.2f}%, Validation accuracy: {:.2f}%'.format(name, train_acc * 100,
                                                                                        val_acc * 100))
    return


def train_and_evaluate_model(name, model, params, train_x, train_y, validation_x, validation_y):
    if params is not None:
        model.set_params(**params)
    model.fit(train_x, train_y)
    evaluate_model(name, model, train_x, train_y, validation_x, validation_y)


def get_svm_model(model_type, kernel_type, max_it):
    if model_type == 'svm':
        if kernel_type == 'linear':
            return LinearSVC(max_iter=max_it)
        else:
            return SVC(kernel=kernel_type, max_iter=max_it)
    else:
        pass


def train_multiple_svm_models_simultaneously(krnl_type, vald_x, vald_y, sub_svm_itr, sub_tr_size,
                                             num_iteration, parameters=None):
    """
    Returns multiple svm models trained simultaneously on different random datasets
    :param krnl_type:
    :param vald_x:
    :param vald_y:
    :param sub_svm_itr:
    :param sub_tr_size:
    :param num_iteration:
    :param parameters:
    :return:
    """
    models = []
    sub_tr_prop = [1, 0, 0]
    for p in range(num_iteration):
        model = get_svm_model('svm', krnl_type, max_it=sub_svm_itr)
        sub_data_set = common_utils.load_random_data_samples(sub_tr_size)
        sub_data_set = common_utils.one_hot_encode(sub_data_set)
        splits = common_utils.split_data_into_train_val_test(sub_data_set, sub_tr_prop)
        sub_tr_x = splits[0]
        sub_tr_y = splits[1]
        train_and_evaluate_model('sub_svm_' + str(p), model, parameters, sub_tr_x, sub_tr_y, vald_x, vald_y)
        models.append(model)
    return models


def train_svm_by_part(krnl_type, vald_x, vald_y, sub_svm_itr, sub_tr_size,
                      num_iteration, params=None):
    """

    :param krnl_type:
    :param vald_x:
    :param vald_y:
    :param sub_svm_itr:
    :param sub_tr_size:
    :param num_iteration:
    :param params:
    :return:
    """
    z = None
    labels = None
    sub_tr_prop = [1, 0, 0]
    for p in range(num_iteration):
        model = get_svm_model('svm', krnl_type, max_it=sub_svm_itr)
        sub_data_set = common_utils.load_random_data_samples(sub_tr_size)
        sub_data_set = common_utils.one_hot_encode(sub_data_set)
        splits = common_utils.split_data_into_train_val_test(sub_data_set, sub_tr_prop)
        sub_tr_x = splits[0]
        sub_tr_y = splits[1]
        print('About to start training')
        train_and_evaluate_model('sub_svm_' + str(p), model, params, sub_tr_x, sub_tr_y, vald_x, vald_y)
        sv = model.support_
        print(f'number of support vectors: {model.n_support_}')
        if z is None:
            z = sub_tr_x[sv]
            labels = sub_tr_y[sv]
        else:
            z = np.concatenate((z, sub_tr_x[sv]), axis=0)
            labels = np.concatenate((labels, sub_tr_y[sv]))
    # now train the master svm
    np.savetxt('all_supp_vectors.txt', np.concatenate((z, labels.reshape(-1, 1)), axis=1))
    model = get_svm_model('svm', krnl_type, max_it=sub_svm_itr)
    train_and_evaluate_model('sub_svm_' + str(p), model, params, z, labels, vald_x, vald_y)
    return model, z, labels


take_majority_vote = True
n_itr = 10000
prop = [0, 1, 1]
test_val_data = 20000  # 20000
one_shot_sample_size = [30000]
number_of_models_to_train = [50]
kernel_type = 'rbf'
c_values = [1] #np.arange(0.1, 1, 0.05)
data_set = common_utils.load_random_data_samples(test_val_data)
data_set = common_utils.one_hot_encode(data_set)
tr_x, tr_y, val_x, val_y, tst_x, tst_y = common_utils.split_data_into_train_val_test(data_set, prop)
print('About to start training')
for i in range(len(one_shot_sample_size)):
    for j in range(len(number_of_models_to_train)):
        for k in range(len(c_values)):
            params = {'C': c_values[k]}
            set_of_svm = train_multiple_svm_models_simultaneously(kernel_type, val_x, val_y, n_itr,
                                                                  one_shot_sample_size[i],
                                                                  number_of_models_to_train[j], params)
            if take_majority_vote:
                print(
                    f'test accuracy of the ensemble is: {100 * majority_voting_model_accuracy(set_of_svm, tst_x, tst_y)}% f'
                    f'or C, tr_size, number pf svms:{c_values[k], one_shot_sample_size[i], number_of_models_to_train[j]}')

# train_svm_by_part(kernel_type, val_x, val_y, n_itr, one_shot_sample_size, 20, params)
# print(f'test accuracy is: {100 * model_accuracy(md, tst_x, tst_y)}%')
# common_utils.save_model_in_pickle('aggregate_model', md)

# print(f'check the training labels: {tr_y}')

# m = get_svm_model('svm', kernel_type, max_it=n_itr)
# model_name = 'Linear SVM'
# print("about to train the model")
# C_values = np.arange(0.1, 5.1, 0.1)
# for i in range(len(C_values)):
#     print(f'Now training with C value:{C_values[i]}')
#     params = {'C': C_values[i]}
#     # start = time.time()
#     train_and_evaluate_mdoel(name=model_name, model=m, params=params, train_x=tr_x, train_y=tr_y,
#                              validation_x=val_x, validation_y=val_y)
#     print(f'test accuracy is: {100 * model_accuracy(m, tst_x, tst_y)}%')
#
# # ================================= try more samples now===============================#


# n_itr = 10000
# prop = [18, 1, 1]
# total_data_size = 100000  # 20000
# kernel_type = 'rbf'
#
# data_set = common_utils.load_random_data_samples(total_data_size)
# data_set = common_utils.one_hot_encode(data_set)
#
# tr_x, tr_y, val_x, val_y, tst_x, tst_y = common_utils.split_data_into_train_val_test(data_set, prop)
#
# # print(f'check the training labels: {tr_y}')
#
#
# m = get_svm_model('svm', kernel_type, max_it=n_itr)
# model_name = 'Linear SVM'
# print("about to train the model")
# C_values = np.arange(1, 200, 30)
# for i in range(len(C_values)):
#     print(f'Now training with C value:{C_values[i]}')
#     params = {'C': C_values[i]}
#     # start = time.time()
#     train_and_evaluate_model(name=model_name, model=m, params=params, train_x=tr_x, train_y=tr_y,
#                              validation_x=val_x, validation_y=val_y)
#     print(f'test accuracy is: {100 * model_accuracy(m, tst_x, tst_y)}%')
