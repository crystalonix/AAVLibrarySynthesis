import matplotlib.pyplot as plt


def plot_histogram(data_dict):
    plt.hist(data_dict)
    plt.show()


def plot_linechart(data_dict, file=None):
    plt.scatter(data_dict.keys(), data_dict.values())
    plt.show()
    plt.savefig(file)
