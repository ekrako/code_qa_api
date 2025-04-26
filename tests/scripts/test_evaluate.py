import subprocess
from pathlib import Path

import pytest

from scripts.evaluate import (
    QA_REPO_URL,
    calculate_similarity,
    clone_qa_repo,
    load_qa_data,
    load_qa_pairs,
)


# Test for calculate_similarity
def test_calculate_similarity_identical():
    """Test that identical sentences have a similarity score of 1.0."""
    text = "This is a test sentence."
    # Need to ensure the model is loaded or mock it
    # For now, assuming direct calculation is feasible in test environment
    score = calculate_similarity(text, text)
    assert score == pytest.approx(1.0)


def test_calculate_similarity_different():
    """Test that very different sentences have a low similarity score."""
    text1 = "The cat sat on the mat."
    text2 = "Quantum physics is complex."
    score = calculate_similarity(text1, text2)
    # The exact score depends on the model, but it should be significantly less than 1
    assert score < 0.5


def test_calculate_similarity_similar():
    """Test that semantically similar sentences have a high similarity score."""
    text1 = "The quick brown fox jumps over the lazy dog."
    text2 = "A fast, dark-colored fox leaps above a sleepy canine."
    score = calculate_similarity(text1, text2)
    # Expecting a relatively high score
    assert score > 0.7


# Tests for load_qa_pairs
def test_load_qa_pairs_success(tmp_path: Path):
    """Test loading valid QA pairs from a directory."""
    # Create dummy files
    q1_path = tmp_path / "1.q.md"
    a1_path = tmp_path / "1.a.md"
    q2_path = tmp_path / "2.q.md"
    a2_path = tmp_path / "2.a.md"
    other_file = tmp_path / "readme.txt"

    q1_path.write_text("Question 1")
    a1_path.write_text("Answer 1")
    q2_path.write_text("Question 2")
    a2_path.write_text("Answer 2")
    other_file.write_text("Ignore me")

    expected_data = [
        {"question": "Question 1", "answer": "Answer 1"},
        {"question": "Question 2", "answer": "Answer 2"},
    ]

    result = load_qa_pairs(tmp_path)
    assert result == expected_data


def test_load_qa_pairs_missing_pair(tmp_path: Path):
    """Test behavior when a question or answer file is missing."""
    q1_path = tmp_path / "1.q.md"
    a2_path = tmp_path / "2.a.md"  # Only answer for 2

    q1_path.write_text("Question 1")
    a2_path.write_text("Answer 2")

    # Expecting only complete pairs to be loaded, or handle warnings/errors
    # Current implementation prints a warning and skips incomplete pairs
    result = load_qa_pairs(tmp_path)
    assert result == []  # No complete pairs found


def test_load_qa_pairs_empty_dir(tmp_path: Path):
    """Test loading from an empty directory."""
    result = load_qa_pairs(tmp_path)
    assert result == []  # Should return an empty list


def test_load_qa_pairs_no_qa_files(tmp_path: Path):
    """Test loading from a directory with no QA files."""
    (tmp_path / "config.yaml").touch()
    (tmp_path / "script.py").touch()

    result = load_qa_pairs(tmp_path)
    assert result == []


def test_load_qa_pairs_invalid_names(tmp_path: Path):
    """Test loading with files having incorrect naming patterns."""
    (tmp_path / "q1.md").write_text("Q1")
    (tmp_path / "a1.md").write_text("A1")
    (tmp_path / "01.question.md").write_text("Q01")
    (tmp_path / "01.answer.md").write_text("A01")

    result = load_qa_pairs(tmp_path)
    assert result == []  # Files don't match the expected pattern


# Tests for load_qa_data
def test_load_qa_data_dir_exists(mocker):
    """Test load_qa_data when the directory already exists."""
    mock_path = mocker.MagicMock(spec=Path)
    mock_path.is_dir.return_value = True
    mock_load_pairs = mocker.patch("scripts.evaluate.load_qa_pairs", return_value=["dummy_data"])
    mock_clone_repo = mocker.patch("scripts.evaluate.clone_qa_repo")

    result = load_qa_data(mock_path)

    mock_path.is_dir.assert_called_once()
    mock_load_pairs.assert_called_once_with(mock_path)
    mock_clone_repo.assert_not_called()
    assert result == ["dummy_data"]


def test_load_qa_data_dir_does_not_exist_clone_success(mocker):
    """Test load_qa_data when dir doesn't exist and cloning succeeds."""
    mock_path = mocker.MagicMock(spec=Path)
    mock_path.is_dir.return_value = False
    # Simulate successful clone
    mock_clone_repo = mocker.patch("scripts.evaluate.clone_qa_repo", return_value=True)
    # Simulate loading data after clone
    mock_load_pairs = mocker.patch("scripts.evaluate.load_qa_pairs", return_value=["cloned_data"])

    result = load_qa_data(mock_path)

    mock_path.is_dir.assert_called_once()
    mock_clone_repo.assert_called_once_with(mock_path)
    mock_load_pairs.assert_called_once_with(mock_path)
    assert result == ["cloned_data"]


def test_load_qa_data_dir_does_not_exist_clone_fail(mocker):
    """Test load_qa_data when dir doesn't exist and cloning fails."""
    mock_path = mocker.MagicMock(spec=Path)
    mock_path.is_dir.return_value = False
    # Simulate failed clone
    mock_clone_repo = mocker.patch("scripts.evaluate.clone_qa_repo", return_value=False)
    mock_load_pairs = mocker.patch("scripts.evaluate.load_qa_pairs")

    # Expect None when cloning fails
    result = load_qa_data(mock_path)

    mock_path.is_dir.assert_called_once()
    mock_clone_repo.assert_called_once_with(mock_path)
    mock_load_pairs.assert_not_called()  # Should not attempt to load if clone failed
    assert result is None


# Tests for clone_qa_repo
def test_clone_qa_repo_success(mocker):
    """Test clone_qa_repo successful execution."""
    mock_path = mocker.MagicMock(spec=Path)
    mock_parent_path = mocker.MagicMock(spec=Path)
    mock_path.parent = mock_parent_path

    mock_run = mocker.patch("subprocess.run")
    mock_run.return_value = subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr="")

    result = clone_qa_repo(mock_path)

    mock_parent_path.mkdir.assert_called_once_with(parents=True, exist_ok=True)
    mock_run.assert_called_once_with(
        ["git", "clone", QA_REPO_URL, str(mock_path)],
        check=True,
        capture_output=True,
        text=True,
    )
    assert result is True


def test_clone_qa_repo_git_fail(mocker):
    """Test clone_qa_repo when git clone command fails."""
    mock_path = mocker.MagicMock(spec=Path)
    mock_parent_path = mocker.MagicMock(spec=Path)
    mock_path.parent = mock_parent_path

    mock_run = mocker.patch("subprocess.run")
    mock_run.side_effect = subprocess.CalledProcessError(returncode=1, cmd=[], stderr="Clone failed")

    result = clone_qa_repo(mock_path)

    mock_parent_path.mkdir.assert_called_once_with(parents=True, exist_ok=True)
    mock_run.assert_called_once()
    assert result is False


def test_clone_qa_repo_mkdir_fail(mocker):
    """Test clone_qa_repo when creating the parent directory fails."""
    mock_path = mocker.MagicMock(spec=Path)
    mock_parent_path = mocker.MagicMock(spec=Path)
    mock_path.parent = mock_parent_path

    mock_parent_path.mkdir.side_effect = OSError("Permission denied")
    mock_run = mocker.patch("subprocess.run")  # Mock run so it doesn't get called

    result = clone_qa_repo(mock_path)

    mock_parent_path.mkdir.assert_called_once_with(parents=True, exist_ok=True)
    mock_run.assert_not_called()
    assert result is False
