"""
base strategy for client selection where in each round all clients available are randomly selected, ignoring battery levels.

For each client if the required energy to perform the training is more than the current battery level, the client drops out and does not complete the training.

"""
