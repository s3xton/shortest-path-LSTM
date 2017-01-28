import sp_lstm
import csv

def write_csv(filename, data, tags):
    with open(filename, 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter=' ',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for i in range(len(tags)):
            writer.writerow(tags[i])
            writer.writerow([data[i]])

def train_suite(output_csv_name, set_size, hyperparams, tags):
    outputs = []
    for i, params in enumerate(hyperparams):
        print("\n#####STARTING " + tags[i] + " #####\n")
        outputs.append(sp_lstm.train_model(set_size,             # Dataset size
                                           params["hidden"],     # Hidden
                                           params["layers"],     # Layers
                                           params["lr"],         # Learning Rate
                                           params["epochs"],     # Epochs
                                           params["batch"],      # Batch Size
                                           params["d_in"],       # Dropout input
                                           params["d_out"]))     # Dropout Output


    write_csv(output_csv_name, outputs, tags)

def train_20k_1layer():
    tags = []
    hyperparams = []

    # Basefor 20k
    tags.append("20-1-base")
    hyperparams.append({"hidden": 200, "layers": 1, "lr": 0.001, "epochs": 50,
                        "batch": 1000, "d_in": 1.0, "d_out": 1.0})

    # High learning rate
    tags.append("20-1-lr-h")
    hyperparams.append({"hidden": 200, "layers": 1, "lr": 0.01, "epochs": 50,
                        "batch": 1000, "d_in": 1.0, "d_out": 1.0})

    # Low learning rate
    tags.append("20-1-lr-l")
    hyperparams.append({"hidden": 200, "layers": 1, "lr": 0.0001, "epochs": 50,
                        "batch": 1000, "d_in": 1.0, "d_out": 1.0})

    # High hidden
    tags.append("20-1-h-h")
    hyperparams.append({"hidden": 300, "layers": 1, "lr": 0.001, "epochs": 50,
                        "batch": 1000, "d_in": 1.0, "d_out": 1.0})

    # Low hidden
    tags.append("20-1-h-l")
    hyperparams.append({"hidden": 100, "layers": 1, "lr": 0.001, "epochs": 50,
                        "batch": 1000, "d_in": 1.0, "d_out": 1.0})

    train_suite("20-1.csv", 20000, hyperparams, tags)


if __name__ == '__main__':
    train_20k_1layer()
