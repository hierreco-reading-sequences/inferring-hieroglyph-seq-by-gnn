from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python < 3.11 fallback guard.
    tomllib = None


CONFIG_ENV_VAR = "HIERRECO_NN_CONFIG"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_FILENAME = "config.toml"


@dataclass(frozen=True)
class ProjectPaths:
    """Filesystem locations used by the command-line tools."""

    dataset_root: Path
    cache_dir: Path
    runs_root: Path
    matplotlib_config_dir: Path


@dataclass(frozen=True)
class ProjectConfig:
    """Resolved project configuration."""

    config_path: Path
    paths: ProjectPaths


DEFAULT_PATHS = {
    "dataset_root": "json_data",
    "cache_dir": "cache",
    "runs_root": "runs",
    "matplotlib_config_dir": ".cache/matplotlib",
}


def _config_path(path: Path | str | None) -> Path:
    if path is not None:
        return Path(path).expanduser().resolve()

    env_path = os.environ.get(CONFIG_ENV_VAR)
    if env_path:
        return Path(env_path).expanduser().resolve()

    cwd_config = Path.cwd() / CONFIG_FILENAME
    if cwd_config.exists():
        return cwd_config.resolve()

    return PROJECT_ROOT / CONFIG_FILENAME


def _read_toml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    if tomllib is None:
        raise RuntimeError("Reading config.toml requires Python 3.11 or newer.")

    with path.open("rb") as file:
        return tomllib.load(file)


def _resolve_path(value: Any, *, base_dir: Path) -> Path:
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    return path


def load_project_config(path: Path | str | None = None) -> ProjectConfig:
    """Load and resolve the project config.

    Relative paths are resolved from the directory containing the config file.
    When no config file exists, defaults are resolved from the repository root.
    """

    env_path = os.environ.get(CONFIG_ENV_VAR)
    config_path = _config_path(path)
    if (path is not None or env_path) and not config_path.exists():
        raise FileNotFoundError(f"Config file does not exist: {config_path}")

    payload = _read_toml(config_path)
    raw_paths = dict(DEFAULT_PATHS)
    raw_paths.update(payload.get("paths", {}))

    base_dir = config_path.parent if config_path.exists() else Path.cwd().resolve()
    paths = ProjectPaths(
        dataset_root=_resolve_path(raw_paths["dataset_root"], base_dir=base_dir),
        cache_dir=_resolve_path(raw_paths["cache_dir"], base_dir=base_dir),
        runs_root=_resolve_path(raw_paths["runs_root"], base_dir=base_dir),
        matplotlib_config_dir=_resolve_path(
            raw_paths["matplotlib_config_dir"],
            base_dir=base_dir,
        ),
    )
    return ProjectConfig(config_path=config_path, paths=paths)


def load_project_config_from_cli(argv: list[str] | None = None) -> ProjectConfig:
    """Pre-parse ``--config`` so argparse defaults can come from that file."""

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config", type=Path, default=None)
    args, _ = parser.parse_known_args(argv)
    return load_project_config(args.config)


def add_config_argument(
    parser: argparse.ArgumentParser,
    config: ProjectConfig,
) -> None:
    """Add the shared config-file argument to a CLI parser."""

    parser.add_argument(
        "--config",
        type=Path,
        default=config.config_path,
        help=(
            "Path to project config TOML. Defaults to config.toml or "
            f"${CONFIG_ENV_VAR}."
        ),
    )


def configure_matplotlib(config: ProjectConfig) -> None:
    """Set a repository-local matplotlib config dir unless the user set one."""

    os.environ.setdefault("MPLCONFIGDIR", str(config.paths.matplotlib_config_dir))
