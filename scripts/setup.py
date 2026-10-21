"""Native Windows/Linux setup. Run with Python 3.12; activation is unnecessary."""
import os
from pathlib import Path
import subprocess
import sys
import venv

ROOT = Path(__file__).resolve().parents[1]


def main():
    if sys.version_info[:2] != (3, 12):
        raise SystemExit(
            "Setup requires Python 3.12. On Windows run: py -3.12 scripts/setup.py. "
            "Install Python 3.12 if the launcher cannot find it."
        )
    environment = ROOT / ".venv"
    python = environment / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    if environment.exists() and not python.is_file():
        raise SystemExit(
            "The existing .venv is incomplete or belongs to another operating system. "
            "Rename it to preserve its contents, then run setup again."
        )
    if not environment.exists():
        venv.EnvBuilder(with_pip=True).create(environment)
    version = subprocess.check_output(
        [str(python), "-c", "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"],
        text=True,
    ).strip()
    if version != "3.12":
        raise SystemExit("The existing .venv is not Python 3.12. Rename it, then rerun setup.")
    env = os.environ.copy()
    env["HF_HUB_DISABLE_XET"] = "1"
    commands = [
        ["pip", "install", "torch==2.6.0", "--index-url", "https://download.pytorch.org/whl/cpu"],
        ["pip", "install", "-r", "requirements.lock"],
        ["scripts.download_muril"],
        ["scripts.verify_artifacts"],
    ]
    for command in commands:
        subprocess.run([str(python), "-m", *command], cwd=ROOT, env=env, check=True)
    print("Setup complete. From the repository directory, start the app with:")
    executable = r".\.venv\Scripts\python.exe" if os.name == "nt" else ".venv/bin/python"
    print(f"{executable} -m uvicorn detector.app:app --host 127.0.0.1 --port 8000")


if __name__ == "__main__":
    main()
