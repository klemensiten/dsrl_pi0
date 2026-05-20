"""Import path bootstrap for local simulation dependencies."""
import sys
from pathlib import Path


def bootstrap_sim_paths():
    repo_root = Path(__file__).resolve().parents[1]
    paths = [
        repo_root / "tactile_envs" / "tactile_envs" / "envs" / "robosuite",
        repo_root / "libero",
    ]
    for path in reversed(paths):
        path_str = str(path)
        if path.exists() and path_str not in sys.path:
            sys.path.insert(0, path_str)
