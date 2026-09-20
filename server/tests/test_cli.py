from pathlib import Path

import pytest
import vsp.cli
from vsp.config import REPO_ROOT

CHECKPOINT = REPO_ROOT / "vision_train" / "sam3_checkpoint_hf"


@pytest.mark.skipif(not CHECKPOINT.is_dir(), reason="converted SAM3 checkpoint not available")
def test_verify_weights_prints_absolute_repo_path_regardless_of_cwd(monkeypatch, tmp_path, capsys):
    monkeypatch.chdir(tmp_path)
    vsp.cli.main(["verify-weights"])
    out = capsys.readouterr().out
    assert str(CHECKPOINT) in out
    assert Path(str(CHECKPOINT)).is_absolute()
