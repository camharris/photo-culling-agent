"""Tests for the Gradio interface."""

import os
import shutil
from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import pytest

from src.photo_culling_agent.gradio_interface.gradio_interface import PhotoCullingInterface

if TYPE_CHECKING:
    from pytest_mock.plugin import MockerFixture


@pytest.fixture
def mock_graph(mocker: "MockerFixture") -> MagicMock:
    """Fixture to mock the PhotoCullingGraph."""
    return mocker.patch(
        "src.photo_culling_agent.gradio_interface.gradio_interface.PhotoCullingGraph"
    )


@pytest.fixture
def interface_instance(mocker: "MockerFixture", tmp_path: str) -> PhotoCullingInterface:
    """Fixture to create an instance of PhotoCullingInterface with mocked dependencies."""
    mocker.patch("os.makedirs")
    mocker.patch("tempfile.mkdtemp", return_value=str(tmp_path / "temp_culling"))
    mock_pipeline_instance = MagicMock()
    mocker.patch(
        "src.photo_culling_agent.gradio_interface.gradio_interface.PhotoCullingGraph",
        return_value=mock_pipeline_instance,
    )

    decision_weights = {"composition": 1.0, "exposure": 1.0}
    interface = PhotoCullingInterface(
        output_dir=str(tmp_path / "output"), decision_weights=decision_weights
    )
    interface.pipeline = mock_pipeline_instance  # Ensure the instance uses the mock
    return interface


class TestPhotoCullingInterface:
    """Test suite for the PhotoCullingInterface class."""

    def test_init(self, mocker: "MockerFixture", tmp_path: str) -> None:
        """Test the initialization of PhotoCullingInterface.

        Args:
            mocker: Pytest mocker fixture.
            tmp_path: Pytest temporary path fixture.
        """
        mock_os_makedirs = mocker.patch("os.makedirs")
        mock_tempfile_mkdtemp = mocker.patch(
            "tempfile.mkdtemp", return_value=str(tmp_path / "test_temp_dir")
        )
        mock_pipeline_init = mocker.patch(
            "src.photo_culling_agent.gradio_interface.gradio_interface.PhotoCullingGraph"
        )

        output_dir = str(tmp_path / "output")
        decision_weights = {"composition": 1.0, "exposure": 1.0}

        interface = PhotoCullingInterface(output_dir=output_dir, decision_weights=decision_weights)

        # Check that output directory is created
        mock_os_makedirs.assert_called_once_with(output_dir, exist_ok=True)

        # Check that temp directory is created
        mock_tempfile_mkdtemp.assert_called_once_with(prefix="photo_culling_")
        assert interface.temp_dir == str(tmp_path / "test_temp_dir")

        # Check that LangGraph pipeline is initialized
        mock_pipeline_init.assert_called_once_with(decision_weights=decision_weights)
        assert interface.pipeline is not None

        # Check that tracking attributes are initialized
        assert interface.processed_images == {}
        assert interface.uploads_in_progress == set()

        # Check that interface is built (minimal check, actual UI testing is harder)
        assert interface.interface is not None

        # Cleanup temp dir created by the actual class constructor if not mocked properly before
        if os.path.exists(interface.temp_dir) and "test_temp_dir" not in interface.temp_dir:
            shutil.rmtree(interface.temp_dir)
        # Cleanup temp dir created by the mock, if it somehow got created (it shouldn't)
        if os.path.exists(str(tmp_path / "test_temp_dir")):
            shutil.rmtree(str(tmp_path / "test_temp_dir"))

    def test_del_cleanup(self, mocker: "MockerFixture", tmp_path: str) -> None:
        """Test the __del__ method for temporary directory cleanup.

        Args:
            mocker: Pytest mocker fixture.
            tmp_path: Pytest temporary path fixture.
        """
        mock_shutil_rmtree = mocker.patch("shutil.rmtree")
        mock_os_path_exists = mocker.patch("os.path.exists", return_value=True)

        # Create an interface instance
        # We need to control the temp_dir attribute for this test
        temp_dir_path = str(tmp_path / "specific_temp_dir_for_del_test")

        # Mock tempfile.mkdtemp specifically for this instance creation
        mocker.patch("tempfile.mkdtemp", return_value=temp_dir_path)
        mocker.patch("os.makedirs")  # Mock makedirs as it's called in init
        mocker.patch(
            "src.photo_culling_agent.gradio_interface.gradio_interface.PhotoCullingGraph"
        )  # Mock graph init

        interface = PhotoCullingInterface(output_dir=str(tmp_path / "output"))
        assert interface.temp_dir == temp_dir_path  # Ensure our mock was used

        # Call __del__
        interface.__del__()

        # Assert that os.path.exists was called with the temp_dir
        mock_os_path_exists.assert_called_with(temp_dir_path)
        # Assert that shutil.rmtree was called with the temp_dir
        mock_shutil_rmtree.assert_called_once_with(temp_dir_path)

    def test_del_cleanup_no_dir(self, mocker: "MockerFixture", tmp_path: str) -> None:
        """Test the __del__ method when the temp directory does not exist.

        Args:
            mocker: Pytest mocker fixture.
            tmp_path: Pytest temporary path fixture.
        """
        mock_shutil_rmtree = mocker.patch("shutil.rmtree")
        mock_os_path_exists = mocker.patch("os.path.exists", return_value=False)

        temp_dir_path = str(tmp_path / "non_existent_temp_dir")
        mocker.patch("tempfile.mkdtemp", return_value=temp_dir_path)
        mocker.patch("os.makedirs")
        mocker.patch("src.photo_culling_agent.gradio_interface.gradio_interface.PhotoCullingGraph")

        interface = PhotoCullingInterface(output_dir=str(tmp_path / "output"))
        interface.__del__()

        mock_os_path_exists.assert_called_with(temp_dir_path)
        mock_shutil_rmtree.assert_not_called()

    def test_del_cleanup_exception(self, mocker: "MockerFixture", tmp_path: str) -> None:
        """Test the __del__ method handles exceptions during rmtree.

        Args:
            mocker: Pytest mocker fixture.
            tmp_path: Pytest temporary path fixture.
        """
        mock_shutil_rmtree = mocker.patch("shutil.rmtree", side_effect=OSError("Test error"))
        mock_os_path_exists = mocker.patch("os.path.exists", return_value=True)
        # mock_print = mocker.patch("builtins.print") # To check if error is printed
        # Patch the logger used in the __del__ method
        mock_logger_error = mocker.patch(
            "src.photo_culling_agent.gradio_interface.gradio_interface.logger.error"
        )

        temp_dir_path = str(tmp_path / "exception_temp_dir")
        mocker.patch("tempfile.mkdtemp", return_value=temp_dir_path)
        mocker.patch("os.makedirs")
        mocker.patch("src.photo_culling_agent.gradio_interface.gradio_interface.PhotoCullingGraph")

        interface = PhotoCullingInterface(output_dir=str(tmp_path / "output"))
        interface.__del__()

        mock_os_path_exists.assert_called_with(temp_dir_path)
        mock_shutil_rmtree.assert_called_once_with(temp_dir_path)
        # mock_print.assert_called_once_with("Error cleaning up temp directory: Test error")
        mock_logger_error.assert_called_once_with("Error cleaning up temp directory: Test error")

    # TODO: Add more tests for handle_upload, analyze_images, show_image_details, export_metadata,
    # and the _get_* helper methods. Consider mocking file operations and pipeline interactions.
