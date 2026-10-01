import pytest
import torch

from landcover.data import synthetic


@pytest.fixture(scope="session")
def paired(tmp_path_factory):
    torch.set_num_threads(2)
    root = tmp_path_factory.mktemp("paired")
    synthetic(root)
    return root
