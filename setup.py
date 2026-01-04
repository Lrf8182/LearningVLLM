from setuptools import setup, find_packages
from pathlib import Path

PKG_NAME = "learning_vllm"
ROOT_DIR = Path(__file__).parent


def get_requirements() -> list[str]:
    """Get Python package dependencies from requirements.txt."""
    requirements_dir = ROOT_DIR / "requirements"

    def _read_requirements(filename: str) -> list[str]:
        with open(requirements_dir / filename) as f:
            requirements = f.read().strip().split("\n")
        resolved_requirements = []
        for line in requirements:
            if line.startswith("-r "):
                resolved_requirements += _read_requirements(line.split()[1])
            elif not line.startswith("--") and not line.startswith("#") and line.strip() != "":
                resolved_requirements.append(line)
        return resolved_requirements

    requirements = _read_requirements("common.txt")
    return requirements


setup(
    package_dir={PKG_NAME: f"./src/{PKG_NAME}"},
    packages=find_packages(where="./src"),
    install_requires=get_requirements(),
)
