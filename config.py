"""Configuration file for the project."""

# Data paths
RAW_DATA_PATH = "data/raw/"
PROCESSED_DATA_PATH = "data/processed/"

# Model hyperparameters
SEQUENCE_LENGTH = 6
MINIMUM_SEQUENCE_LENGTH = 3
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.001
VALIDATION_SPLIT = 0.2
TEST_SPLIT = 0.1

# MODEL_TYPE = 'lstm'
# MODEL_TYPE = 'transformer'
# MODEL_TYPE = 'attention_lstm'
# MODEL_TYPE = 'bigru'

# LSTM parameters
LSTM_UNITS = 128

# Transformer parameters
TRANSFORMER_HEADS = 4
TRANSFORMER_DIM = 128
TRANSFORMER_FF_DIM = 512
TRANSFORMER_BLOCKS = 2

# Dropout rate
DROPOUT = 0.2

# Training callbacks configuration
EARLY_STOPPING_PATIENCE = 10
REDUCE_LR_PATIENCE = 5
REDUCE_LR_FACTOR = 0.5
MIN_LEARNING_RATE = 1e-7

