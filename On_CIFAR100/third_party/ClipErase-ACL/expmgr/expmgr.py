import yaml
import logging
from datetime import datetime

from pathlib import Path

logging.basicConfig(level=logging.INFO)


class ExperimentManager:
    def __init__(self) -> None:
        self.settings = None
        self.time_stamp = datetime.now()
        self.time_stemp_str: str = self.time_stamp.strftime(
            "%Y-%m-%d-%H-%M-%S")

    def __enter__(self) -> dict:
        # Read settings
        settings_file_path = Path("./expmgr/template_settings.yml")
        if Path("./user_settings.yml").exists():
            settings_file_path = Path("./user_settings.yml")

        settings = {}
        with open(settings_file_path, 'r') as f:
            settings = yaml.load(f, Loader=yaml.FullLoader)

        recent_experiments_path: Path = Path("./expmgr/recent_experiments/")
        if not recent_experiments_path.exists():
            recent_experiments_path.mkdir()

        all_tensorboard_dir: Path = Path("./expmgr/tensorboard")
        if not all_tensorboard_dir.exists():
            all_tensorboard_dir.mkdir()

        all_checkpoints_dir: Path = Path("./expmgr/checkpoints")
        if not all_checkpoints_dir.exists():
            all_checkpoints_dir.mkdir()

        all_heatmaps_dir: Path = Path("./expmgr/heatmaps")
        if not all_heatmaps_dir.exists():
            all_heatmaps_dir.mkdir()

        # Assign tensorboard path
        if settings["mode"] == "train":
            self.assign_tensorboard_path(settings)

        # Assign checkpoint path
        if settings["mode"] == "train":
            self.assign_checkpoint_path(settings)

        # Assign heatmap path
        if settings["mode"] == "train":
            self.assign_heatmap_path(settings)

        self.settings = settings

        print(f"Tensorboard dir for this event: {settings['train']['tensorboard']}. Please run \"tensorboard --logdir {settings['train']['tensorboard']}\" to check the experiment status.")

        return settings

    def __exit__(self, exc_type, exc_value, traceback):
        recent_experiments_path: Path = Path("./expmgr/recent_experiments/")
        if not recent_experiments_path.exists():
            recent_experiments_path.mkdir()
        file_path: Path = recent_experiments_path / \
            (self.time_stemp_str + ".yml")
        with open(file_path, "w") as f:
            yaml.dump(self.settings, f, default_flow_style=False)
            logging.info(f"File {file_path.as_posix()} has been saved.")
        return

    def assign_tensorboard_path(self, settings) -> None:
        tensorboard_dir: Path = Path(
            "./expmgr/tensorboard") / self.time_stemp_str
        if not tensorboard_dir.exists():
            tensorboard_dir.mkdir()

        settings["train"]["tensorboard"] = tensorboard_dir.as_posix()
        return

    def assign_checkpoint_path(self, settings) -> None:
        checkpoints_dir: Path = Path(
            "./expmgr/checkpoints") / self.time_stemp_str
        if not checkpoints_dir.exists():
            checkpoints_dir.mkdir()

        settings["train"]["checkpoint"] = checkpoints_dir.as_posix()
        return

    def assign_heatmap_path(self, settings) -> None:
        heatmaps_dir: Path = Path(
            "./expmgr/heatmaps") / self.time_stemp_str
        if not heatmaps_dir.exists():
            heatmaps_dir.mkdir()

        settings["train"]["heatmap"] = heatmaps_dir.as_posix()
        return