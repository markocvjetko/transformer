
import subprocess
from lightning import Callback


class GitCallback(Callback):
    """
    Saves a git hash once training starts. To load the saved commit, do "git checkout $(cat path/to/hash)"
    To go back to the latest commit, "git pull origin branchname"
    """
    def __init__(self, experiment_root) -> None:
        super().__init__()
        self.experiment_root = experiment_root

    def on_fit_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        
        with open(self.experiment_root / "git_hash", mode="w") as f:
            f.write(subprocess.check_output(["git", "rev-parse", "HEAD"], text=True))