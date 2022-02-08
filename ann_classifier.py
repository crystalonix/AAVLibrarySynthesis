import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
from sklearn.svm import LinearSVC, SVC
from sklearn.model_selection import GridSearchCV
import common_utils
import plot_utils
from sklearn.base import clone
import pickle
from matplotlib import pyplot as plt


# Let's create a custom callback class
class PerfEvalCustomCallback(keras.callbacks.Callback):

    def __init__(self, perf_data):
        self.perf_data = perf_data

    # we define the on_epoch_end callback and save the loss and accuracy in perf_data
    def on_epoch_end(self, epoch, logs=None):
        self.perf_data[epoch, 0] = logs['loss']
        self.perf_data[epoch, 1] = logs['accuracy']
        self.perf_data[epoch, 2] = logs['val_loss']
        self.perf_data[epoch, 3] = logs['val_accuracy']

    def get_perf_data(self):
        return self.perf_data


# Customize this function as you like but makes sure it is implemented correctly.
# Note: If you need to change the method definition to add more arguments, make sure to make
# the new arguments are optional (& have a sensible default value)
def evaluate_model(name, model, eval_data,
                   plot_training=True, evaluate_on_test_set=True):
    # unpack the stuff
    perf_data, dataset = eval_data
    train_x, train_y, val_x, val_y, test_x, test_y = dataset

    # get predictions from the model
    train_preds = model.predict(train_x)
    val_preds = model.predict(val_x)

    # measure the accuracy (as categorical accuracy since we have a softmax layer)
    catacc_metric = keras.metrics.BinaryAccuracy()
    catacc_metric.update_state(train_y, train_preds)
    train_acc = catacc_metric.result()

    catacc_metric = keras.metrics.BinaryAccuracy()
    catacc_metric.update_state(val_y, val_preds)
    val_acc = catacc_metric.result()
    print('[{}] Training Accuracy: {:.3f}%, Validation Accuracy: {:.3f}%'.format(name, 100 * train_acc, 100 * val_acc))

    if plot_training:
        plot_training_perf(perf_data[:, 0], perf_data[:, 1], perf_data[:, 2], perf_data[:, 3])

    if evaluate_on_test_set:
        ### Evaluate the model on the test data  and put the results in 'test_loss', 'test_acc' (set verbose=0)
        ###* put your code here (~1-2 lines) *###
        test_loss, test_acc = model.evaluate(test_x, test_y, verbose=0)

        print('[{}] Test loss: {:.5f}, test accuracy: {:.3f}%'.format(name, test_loss, 100 * test_acc))

    # You can add stuff here
    ###* put your code here (0+ lines) *###

    return


# Plot the model's performance during training (across epochs)
def plot_training_perf(train_loss, train_acc, val_loss, val_acc, fs=(8, 5)):
    plt.figure(figsize=fs)

    assert train_loss.shape == val_loss.shape and train_loss.shape == val_acc.shape and val_acc.shape == train_acc.shape

    # assume we have one measurement per epoch
    num_epochs = train_loss.shape[0]
    epochs = np.arange(0, num_epochs)

    # Can you figure out why this makes sense? Why remove -0.5?
    plt.plot(epochs - 0.5, train_loss, 'm', linewidth=2, label='Loss (Training)')
    plt.plot(epochs - 0.5, train_acc, 'r--', linewidth=2, label='Accuracy (Training)')

    plt.plot(epochs, val_loss, 'g', linewidth=2, label='Loss (Validation)')
    plt.plot(epochs, val_acc, 'b:', linewidth=2, label='Accuracy (Validation)')

    plt.xlim([0, num_epochs])
    plt.ylim([0, 1.05])

    plt.legend()

    plt.show()


def train_model(model, dataset, max_epochs=25, batch_size=100, verbose=0,
                ):
    # unpack dataset
    train_x, train_y, val_x, val_y, test_x, test_y = dataset

    # this is the callback we'll use for early stopping
    early_stop_cb = keras.callbacks.EarlyStopping(monitor='loss', mode='min', patience=4)

    # setup the performance data callback
    perf_data = np.zeros((max_epochs, 4))
    perf_eval_cb = PerfEvalCustomCallback(perf_data)

    hobj = model.fit(train_x, train_y, validation_data=(val_x, val_y), epochs=max_epochs, batch_size=batch_size,
                     shuffle=True, callbacks=[perf_eval_cb, early_stop_cb], verbose=verbose)

    eff_epochs = len(hobj.history['loss'])
    eval_data = (perf_data[0:eff_epochs, :], dataset)  # tuple of evaluation data

    return eval_data


def create_compile_model0(fixed, kernel_penalty, bias_penalty, input_shape=858, hidden_widths=[300, 100], num_outputs=1,
                          verbose=True):
    name = 'Model0--Fixed' if fixed else 'Model0--Broken'
    model = keras.models.Sequential(name=name)

    model.add(keras.Input(shape=(input_shape,), sparse=False))

    for i, hw in enumerate(hidden_widths):
        model.add(
            keras.layers.Dense(hw, kernel_regularizer=keras.regularizers.l2(kernel_penalty),
                               bias_regularizer=keras.regularizers.l2(bias_penalty),
                               activation='relu', name='hidden_{}'.format(i),
                               kernel_initializer=keras.initializers.RandomNormal(stddev=np.sqrt(1 / hw)),
                               bias_initializer=keras.initializers.Zeros()))

    model.add(keras.layers.Dense(num_outputs, kernel_regularizer=keras.regularizers.l2(kernel_penalty),
                                 bias_regularizer=keras.regularizers.l2(bias_penalty), activation='sigmoid',
                                 name='output',
                                 kernel_initializer=keras.initializers.RandomNormal(stddev=np.sqrt(0.1)),
                                 bias_initializer=keras.initializers.Zeros()))

    opt = keras.optimizers.Adam(lr=0.0025)

    if verbose:
        model.summary()

    if fixed:
        ###* put your code here (~1-2 lines) *###
        model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
        # comment/remove this line once you implement the fix
        # raise NotImplementedError

    else:
        model.compile(loss='mse', optimizer=opt, metrics=['accuracy'])

    return name, model


l2_kernel_penalty = 0.001  # 0.001
l2_bias_penalty = 0.001  # 0.001
prop = [7, 2, 1]
data_set_size = 30000  # 20000
threshold_dist = 6
itr_no = 5
filter = False

for j in range(itr_no):
    data_set, dists = common_utils.load_random_data_samples_with_mutation_dist(data_set_size)
    if filter:
        data_set, dists = common_utils.filter_data_points_on_mutation_distance(data_set, dists, threshold_dist)
    print(f'initial distance distribution: {common_utils.convert_dist_to_dictionary(dists)}')
    # plot_utils.plot_histogram(dists)
    # print(f'check the distances from wild type: {np.sort(dists)}')
    print(f'number of points within threshold distance: {len(data_set)}')

    data_set = common_utils.one_hot_encode(data_set)
    tr_x, tr_y, val_x, val_y, tst_x, tst_y, dists_tr, dists_val, dists_tst = \
        common_utils.split_data_into_train_val_test(data_set, prop, dists=dists)
    dataset = (tr_x, tr_y, val_x, val_y, tst_x, tst_y)
    fixed = True
    # hdn_wdths = [300, 100]
    # name1, model1 = create_compile_model0(fixed, hidden_widths=hdn_wdths, verbose=True)
    # eval_data = train_model(model1)
    # evaluate_model(name1, model1, eval_data)
    x = 800
    n = 3
    hdn_wdths = [x for i in range(n)]
    name2, model2 = create_compile_model0(fixed, l2_kernel_penalty, l2_bias_penalty, hidden_widths=hdn_wdths,
                                          verbose=True)
    eval_data = train_model(model2, dataset)
    evaluate_model(name2, model2, eval_data, plot_training=False)
    tst_pred = model2.predict(tst_x)
    # accurate_tst_data = tst_x[tst_pred == tst_y]
    accurate_dists = dists_tst[(np.ravel(tst_pred) > 0.5) == tst_y]
    accr_dict = common_utils.convert_dist_to_dictionary(accurate_dists)
    dists_tst_dict = common_utils.convert_dist_to_dictionary(dists_tst)
    for k in dists_tst_dict:
        if k in accr_dict:
            accr_dict[k] = accr_dict[k] / dists_tst_dict[k]
        else:
            print(f'not found in itr:{j}, dist:{dists_tst_dict[k]}')
            accr_dict[k] = 0
    # # inaccurate_test_data = tst_x[tst_pred != tst_y, :]
    # inaccurate_dists = dists_tst[(np.ravel(tst_pred) > 0.5) != tst_y]
    with open('data-dist' + str(j) + '-' + str(data_set_size) + '.txt', 'w') as file:
        file.write(str(dists_tst_dict))
    with open('accuracy-dist' + str(j) + '-' + str(data_set_size) + '.txt', 'w') as file:
        file.write(str(accr_dict))
    print(f' distribution of accurate predictions: {common_utils.convert_dist_to_dictionary(accurate_dists)}')
    # print(f' distribution of inaccurate predictions: {common_utils.convert_dist_to_dictionary(inaccurate_dists)}')
