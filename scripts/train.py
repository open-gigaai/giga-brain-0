import os
import sys
from pathlib import Path

import tyro

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
REPO_ROOT = PROJECT_ROOT.parents[2]
CODE_ROOT = REPO_ROOT.parent


def _resolve_dependency_root(env_name: str, *sibling_names: str) -> Path:
    configured = os.environ.get(env_name)
    if configured:
        root = Path(configured).expanduser().resolve()
        if not root.is_dir():
            raise FileNotFoundError(f'{env_name} does not exist: {root}')
        return root
    for sibling_name in sibling_names:
        root = CODE_ROOT / sibling_name
        if root.is_dir():
            return root
    expected = ', '.join(os.fspath(CODE_ROOT / name) for name in sibling_names)
    raise FileNotFoundError(f'Set {env_name}; no dependency checkout found at: {expected}')


GIGA_TRAIN_ROOT = _resolve_dependency_root('GIGA_TRAIN_ROOT', 'giga-train')
GIGA_DATASETS_ROOT = _resolve_dependency_root(
    'GIGA_DATASETS_ROOT', 'giga-datasets', 'giga-datasets-v3.0'
)
os.environ['GIGA_TRAIN_ROOT'] = os.fspath(GIGA_TRAIN_ROOT)
os.environ['GIGA_DATASETS_ROOT'] = os.fspath(GIGA_DATASETS_ROOT)
BOOTSTRAP_PATHS = tuple(
    os.fspath(path)
    for path in (
        REPO_ROOT,
        PROJECT_ROOT,
        GIGA_TRAIN_ROOT,
        GIGA_DATASETS_ROOT,
    )
)
for _path in reversed(BOOTSTRAP_PATHS):
    if _path not in sys.path:
        sys.path.insert(0, _path)

# Accelerate launches fresh Python processes, so make the same imports visible
# outside this parent interpreter as well.
_inherited_pythonpath = [
    path for path in os.environ.get('PYTHONPATH', '').split(os.pathsep) if path
]
os.environ['PYTHONPATH'] = os.pathsep.join(
    dict.fromkeys((*BOOTSTRAP_PATHS, *_inherited_pythonpath))
)

from giga_train import launch_from_config, setup_environment  # noqa: E402


def train(config: str):
    setup_environment()
    launch_from_config(config)


if __name__ == '__main__':
    tyro.cli(train)
