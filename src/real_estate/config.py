from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ProjectPaths:
    root: Path
    artifacts: Path

    @staticmethod
    def default() -> "ProjectPaths":
        root = Path.cwd()
        artifacts = root / "artifacts"
        artifacts.mkdir(parents=True, exist_ok=True)
        return ProjectPaths(root=root, artifacts=artifacts)


@dataclass(frozen=True)
class TrainingConfig:
    test_size: float = 0.2
    random_state: int = 42
    n_estimators: int = 300
    max_depth: int | None = None
    min_samples_leaf: int = 2
    n_jobs: int = -1
