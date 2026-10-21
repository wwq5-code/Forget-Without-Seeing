import shutil
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)


def setup() -> None:
    if (Path.cwd() / "user_settings.yml").exists():
        while True:
            ans: str = input(
                "Do you want to overwrite the current 'user_settings.yml' file? (y / n) ")
            if ans == "y":
                shutil.copyfile(
                    "./expmgr/template_settings.yml",
                    "./user_settings.yml")
                logging.info("'user_settings.yml' regenerated.")
                break
            elif ans == "n":
                break
            else:
                pass

    else:
        shutil.copyfile(
            "./expmgr/template_settings.yml",
            "./user_settings.yml")

    recent_experiments_path: Path = Path("./expmgr/recent_experiments/")
    if not recent_experiments_path.exists():
        recent_experiments_path.mkdir()

    all_tensorboard_dir: Path = Path("./expmgr/tensorboard")
    if not all_tensorboard_dir.exists():
        all_tensorboard_dir.mkdir()

    all_checkpoints_dir: Path = Path("./expmgr/checkpoints")
    if not all_checkpoints_dir.exists():
        all_checkpoints_dir.mkdir()
    return


if __name__ == "__main__":
    setup()
