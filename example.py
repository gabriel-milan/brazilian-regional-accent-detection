# Import stuff
from training import *

# Load envs
load_environments()

# Train using example config and model
train_with_config("train_config_example.json", "model_example.json")
