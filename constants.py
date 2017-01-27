"""
Place to hold all the constants used.
"""
# Model parameters
INPUT_SIZE = 22
OUTPUT_SIZE = 20
QUERY_PHASE_LENGTH = 1

# Hyperparameters
LEARNING_RATE = 0.01
OPTIMIZER = "ADAM"
NUM_HIDDEN = 200
PLAN_PHASE_LENGTH = 10
DROPOUT_INPUT = 1.0 # between 0 and 1
DROPOUT_OUTPUT = 0.5 # between 0 and 1

# Training data parameters
# max so far 50k @ 3 layers, 200 hidden
DATASET_SIZE = 500000
BATCH_SIZE = 1000
NUM_EPOCHS = 10

# Graph parameters
MAX_NODE = 10
MIN_NODE = 6
