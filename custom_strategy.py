from typing import Iterable
from flwr.serverapp import Grid
from flwr.serverapp.strategy import FedAdagrad
from flwr.app import ArrayRecord, ConfigRecord, Message


class CustomFedAdagrad(FedAdagrad):
    def configure_train(
        self, server_round: int, arrays: ArrayRecord, config: ConfigRecord, grid: Grid
    ) -> Iterable[Message]:
        """Configure next round of training with LR decay"""
        if server_round % 5 == 0 and server_round > 0:
            config["learning-rate"] *= 0.5
            print("LR decreased to:", config["learning-rate"])
        return super().configure_train(server_round, arrays, config, grid)

