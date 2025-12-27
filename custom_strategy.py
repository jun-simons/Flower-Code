from typing import Iterable
from flwr.serverapp import Grid
from flwr.serverapp.strategy import FedAdagrad
from flwr.app import ArrayRecord, ConfigRecord, Message


class CustomFedAdagrad(FedAdagrad):
    def configure_train():
