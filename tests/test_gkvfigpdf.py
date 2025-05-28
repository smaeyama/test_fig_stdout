from pathlib import Path
import pytest
from gkvfigpdf import gkvfigpdf

@pytest.fixture
def sample_log_dir() -> Path:
    """Return the path to the test_data directory with real-looking structure."""
    return Path(__file__).parent / "test_data"

def test_gkvfigpdf_success(sample_log_dir: Path, tmp_path: Path, monkeypatch):
    """Test that gkvfigpdf successfully runs and produces the final PDF output."""
    monkeypatch.chdir(tmp_path)  # Change working directory to tmp_path

    gkvfigpdf(sample_log_dir)

    output_dirs = list(tmp_path.glob("figpdf_*"))
    assert output_dirs, "No output directory was created"

    final_pdf = output_dirs[0] / "fig_stdout.pdf"
    assert final_pdf.exists(), "final_output.pdf was not created"

def test_gkvfigpdf_invalid_dir():
    """Test that gkvfigpdf raises an error when given a nonexistent path."""
    with pytest.raises(FileNotFoundError):
        gkvfigpdf("/non/existent/path")

