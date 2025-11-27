import tyro
from giga_train import launch_from_config, setup_environment


def train(config: str):
    setup_environment()
    launch_from_config(config)


if __name__ == '__main__':
    tyro.cli(train)
