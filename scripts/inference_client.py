import time

import torch
import tyro
from giga_models.sockets import RobotInferenceClient


def _random_observation() -> dict:
    return {
        'observation.state': torch.ones((14,)),
        'observation.images.cam_high': torch.randint(256, size=(3, 224, 224), dtype=torch.uint8) * 1.0 / 255.0,
        'observation.images.cam_left_wrist': torch.randint(256, size=(3, 224, 224), dtype=torch.uint8) * 1.0 / 255.0,
        'observation.images.cam_right_wrist': torch.randint(256, size=(3, 224, 224), dtype=torch.uint8) * 1.0 / 255.0,
        'task': 'task name here',
    }


def run_client(
    host: str = '127.0.0.1',
    port: int = 8080,
):
    """Run the GigaBrain-0 inference client for robot control.

    Args:
        host: Inference server host address.
        port: Inference server port number.
    """
    client = RobotInferenceClient(host=host, port=port)

    while True:
        actions = model_inference(client)
        print(f'Actions: {actions}')
        time.sleep(1)


def model_inference(client: RobotInferenceClient):
    """Run the model inference.

    Args:
        client: The client for communicating with the inference server.
    """
    actions = client.inference(_random_observation())
    return actions.float().cpu().numpy()


if __name__ == '__main__':
    tyro.cli(run_client)
