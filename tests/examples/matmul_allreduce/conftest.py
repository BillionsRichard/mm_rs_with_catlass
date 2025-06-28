import pytest
import os


def pytest_addoption(parser):
    "Pytest hook to add command line options."
    parser.addoption(
        "--executable_path",
        action="store",
        default="tests/bin/matmul_allreduce",
        help="Path to the matmul_allreduce executable",
    )
    parser.addoption(
        "--test_data_dir",
        action="store",
        default="tests/test_data/matmul_allreduce",
        help="Directory to store persistent test data",
    )


@pytest.fixture
def executable_path(request):
    "Fixture for the matmul_allreduce executable path."
    return request.config.getoption("--executable_path")


@pytest.fixture
def test_data_dir(request):
    "Fixture for the test data directory."
    path = request.config.getoption("--test_data_dir")
    os.makedirs(path, exist_ok=True)
    return path
