"""
Module used to train the network with hyperparameters settings presented in paper.
Includes functions for storing and plotting the results.
"""
import csv
import os
import sp_lstm
import matplotlib.pyplot as plt

def read_csv(filename):
    """
    Function for reading results from a csv file

    Args:
        filename: The csv file to be read
    Returns:
        output: The results contained in the csv file in the form of a list
    """
    output = []
    with open(os.path.join("outputs", filename), 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=' ',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for row in reader:
            output.append(row)

    return output


def write_csv(filename, data):
    """
    Function for writing results to a csv file.

    Args:
        filename: The name of the csv file to be created/updated.
        data: The results of training in list form
    """
    with open(os.path.join("outputs", filename), 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter=' ',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for value in data:
            writer.writerow(value)

def train_suite(config_list, tags):
    """
    Function to perform a series of trainings with different hyperparamter configurations

    Args:
        config_list: A list of hyperparameter configuration dictionaries.
        tags: A tag to differentiate each configuration
    """
    outputs = []
    for i, config in enumerate(config_list):
        print("\n#####STARTING " + tags[i] + " #####\n")
        outputs.append(sp_lstm.train_model(config["set_size"],   # Dataset size
                                           config["min_node"],   # Minimum graph size in dataset
                                           config["max_node"],   # Maximum graph size in dataset
                                           config["hidden"],     # Number of hidden units
                                           config["layers"],     # Number of layers
                                           config["lr"],         # Learning rate
                                           config["epochs"],     # Number of epochs
                                           config["batch"],      # Batch size
                                           config["d_in"],       # Dropout input probability
                                           config["d_out"]))     # Dropout output probability

        # Write a file for each config incase one of them crashes
        write_csv(tags[i] + ".csv", outputs[i])

def train_20k_1layer():
    """
    Trains the network with one layer and various configurations on a dataset of size 20,000
    """
    tags = []
    config_list = []

    # Base for 20k 1 layer
    tags.append("20-1-base")
    config_list.append({"set_size": 20000, "min_node": 2, "max_node": 10, "hidden": 200,
                        "layers": 1, "lr": 0.001, "epochs": 50,
                        "batch": 1000, "d_in": 1.0, "d_out": 1.0})

    # High learning rate
    tags.append("20-1-lr-h")
    config_list.append({"set_size": 20000, "min_node": 2, "max_node": 10, "hidden": 200,
                        "layers": 1, "lr": 0.01, "epochs": 50,
                        "batch": 1000, "d_in": 1.0, "d_out": 1.0})

    # Low learning rate
    tags.append("20-1-lr-l")
    config_list.append({"set_size": 20000, "min_node": 2, "max_node": 10, "hidden": 200,
                        "layers": 1, "lr": 0.0001, "epochs": 50,
                        "batch": 1000, "d_in": 1.0, "d_out": 1.0})

    # High hidden
    tags.append("20-1-h-h")
    config_list.append({"set_size": 20000, "min_node": 2, "max_node": 10, "hidden": 300,
                        "layers": 1, "lr": 0.001, "epochs": 50,
                        "batch": 1000, "d_in": 1.0, "d_out": 1.0})

    # Low hidden
    tags.append("20-1-h-l")
    config_list.append({"set_size": 20000, "min_node": 2, "max_node": 10, "hidden": 100,
                        "layers": 1, "lr": 0.001, "epochs": 50,
                        "batch": 1000, "d_in": 1.0, "d_out": 1.0})

    train_suite(config_list, tags)

def train_20k_3layer():
    """
    Trains the network with three layers and various configurations on a dataset of size 20,000
    """
    tags = []
    config_list = []

    # Base for 20k, reduced batch size because GPU can't handle 1000
    tags.append("20-3-base")
    config_list.append({"set_size": 20000, "min_node": 2, "max_node": 10, "hidden": 200,
                        "layers": 3, "lr": 0.001, "epochs": 50,
                        "batch": 800, "d_in": 1.0, "d_out": 1.0})

    # Used to compare effect of batch size with same layers
    tags.append("20-3-base-600B")
    config_list.append({"set_size": 20000, "min_node": 2, "max_node": 10, "hidden": 200,
                        "layers": 3, "lr": 0.001, "epochs": 50,
                        "batch": 600, "d_in": 1.0, "d_out": 1.0})

    # Used to compare effect of batch size accross runs with different layers
    tags.append("20-1-base-800B")
    config_list.append({"set_size": 20000, "min_node": 2, "max_node": 10, "hidden": 200,
                        "layers": 1, "lr": 0.001, "epochs": 50,
                        "batch": 800, "d_in": 1.0, "d_out": 1.0})

    # High learning rate
    tags.append("20-3-lr-h")
    config_list.append({"set_size": 20000, "min_node": 2, "max_node": 10, "hidden": 200,
                        "layers": 3, "lr": 0.01, "epochs": 50,
                        "batch": 800, "d_in": 1.0, "d_out": 1.0})

    # Low learning rate
    tags.append("20-3-lr-l")
    config_list.append({"set_size": 20000, "min_node": 2, "max_node": 10, "hidden": 200,
                        "layers": 3, "lr": 0.0001, "epochs": 50,
                        "batch": 800, "d_in": 1.0, "d_out": 1.0})

    # Dropout output 0.5
    tags.append("20-3-do-50")
    config_list.append({"set_size": 20000, "min_node": 2, "max_node": 10, "hidden": 200,
                        "layers": 3, "lr": 0.001, "epochs": 50,
                        "batch": 600, "d_in": 1.0, "d_out": 0.5})

    # Dropout output 0.8
    tags.append("20-3-do-80")
    config_list.append({"set_size": 20000, "min_node": 2, "max_node": 10, "hidden": 200,
                        "layers": 3, "lr": 0.001, "epochs": 50,
                        "batch": 600, "d_in": 1.0, "d_out": 0.8})

    train_suite(config_list, tags)

def train_20k_vary_size():
    """
    Trains the network with the same configuration on datasets (size 20,000)
    of increasing graph size (2 to 10 vertices inclusive).
    """
    config_list = []
    tags = []
    for i in range(2, 11):
        config_list.append({"set_size": 20000, "min_node": i, "max_node": i, "hidden": 200,
                            "layers": 1, "lr": 0.01, "epochs": 50,
                            "batch": 1000, "d_in": 1.0, "d_out": 1.0})
        tags.append("single - {0}".format(i))

    train_suite(config_list, tags)

def triple_plot(outputs, labels, title, x_label, y_label):
    """
    Bit hacky. Plots two or three lines on the same graph.
    """
    x = range(len(outputs[0]))
    y1 = outputs[0]
    y2 = outputs[1]

    plt.plot(x, y1, ':r', label=labels[0])
    plt.plot(x, y2, '-x', label=labels[1])
    if len(labels) > 2:
        y3 = outputs[2]
        plt.plot(x, y3, '--g', label=labels[2])
    plt.ylim(0, 100)
    #plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()
    plt.show()

def plot_outputs():
    """
    Generates all the plots used in the paper.
    """
    x_label = "Epoch"
    y_label = "Error Rate"
    # Plot 1 - Learning rate, 3 layer
    outputs = []
    labels = []
    outputs.append(read_csv("20-3-base.csv"))
    labels.append("0.001")
    outputs.append(read_csv("20-3-lr-h.csv"))
    labels.append("0.01")
    outputs.append(read_csv("20-3-lr-l.csv"))
    labels.append("0.001")

    triple_plot(outputs, labels, "3 - LR", x_label, y_label)

    # Plot 2 - Dropout, 3 layer
    outputs = []
    labels = []
    outputs.append(read_csv("20-3-base-600B.csv"))
    labels.append("100%")
    outputs.append(read_csv("20-3-do-80.csv"))
    labels.append("80%")
    outputs.append(read_csv("20-3-do-50.csv"))
    labels.append("50%")

    triple_plot(outputs, labels, "3 - Dropout", x_label, y_label)

    # Plot 3 - Learning Rate, 1 layer
    outputs = []
    labels = []
    outputs.append(read_csv("20-1-base.csv"))
    labels.append("0.001")
    outputs.append(read_csv("20-1-lr-h.csv"))
    labels.append("0.01")
    outputs.append(read_csv("20-1-lr-l.csv"))
    labels.append("0.0001")

    triple_plot(outputs, labels, "1 - LR", x_label, y_label)

    # Plot 4 - Hidden, 1 layer
    outputs = []
    labels = []
    outputs.append(read_csv("20-1-base.csv"))
    labels.append("200")
    outputs.append(read_csv("20-1-h-h.csv"))
    labels.append("300")
    outputs.append(read_csv("20-1-h-l.csv"))
    labels.append("100")

    triple_plot(outputs, labels, "1 - Hidden", x_label, y_label)

    # Plot 5 - 1 Layer v 3 Layer
    outputs = []
    labels = []
    outputs.append(read_csv("20-1-base-800B.csv"))
    labels.append("1-Layer")
    outputs.append(read_csv("20-3-base.csv"))
    labels.append("3-Layer")

    triple_plot(outputs, labels, "1 vs 3 layer", x_label, y_label)

    # Plot 6 - Error v Graph size
    outputs = []
    for i in range(2, 11):
        outputs.append(read_csv("single - {0}.csv".format(i))[-1])

    plt.plot(list(range(2, 11)), outputs, ':r')
    plt.ylim(0, 100)
    plt.xlabel("Number of nodes")
    plt.ylabel("Error Rate")
    plt.show()

if __name__ == '__main__':
    train_20k_1layer()
    train_20k_3layer()
    train_20k_vary_size()
    plot_outputs()

