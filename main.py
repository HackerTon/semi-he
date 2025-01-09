import argparse
from pathlib import Path

from src.service.parameter import Parameter
from src.service.trainer_dirichlet import TrainerDirichlet


def run(namespace: argparse.Namespace):
    if not Path(namespace.path).exists():
        print(f"Dataset not found in '{namespace.path}'")
        return

    parameter = Parameter()
    parameter.epoch = namespace.epoch
    parameter.name = namespace.name
    parameter.batch_size_test = namespace.batchsize
    parameter.batch_size_train = namespace.batchsize
    parameter.learning_rate = namespace.learning_rate
    parameter.data_path = namespace.path
    parameter.step = namespace.step
    parameter.train_report_rate = 0.1
    parameter.device = namespace.mode
    parameter.pretrain_path = namespace.pretrain_path

    trainer = TrainerDirichlet(parameter)
    trainer.train()


if __name__ == "__main__":
    parser: argparse.ArgumentParser = argparse.ArgumentParser()
    parser.add_argument("-e", "--epoch", default=50, type=int)
    parser.add_argument("-m", "--mode", default="cpu", type=str)
    parser.add_argument("-b", "--batchsize", default=1, type=int)
    parser.add_argument("-p", "--path", required=True, type=str)
    parser.add_argument("-l", "--learning_rate", default=0.001, type=float)
    parser.add_argument("--pretrain_path", required=True)
    parser.add_argument(
        "-n",
        "--name",
        required=True,
        type=str,
        help="Name of the experiment",
    )
    parser.add_argument(
        "-s",
        "--step",
        type=int,
        default=0,
        help="Step in your experiment",
    )

    parsed_data = parser.parse_args()
    run(parsed_data)
