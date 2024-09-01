import glob
import os
from typing import List


def get_all_file_paths(directory: str, extension: str) -> List[str]:
    """Get all file paths with a given extension from a directory."""
    return glob.glob(os.path.join(directory, f"*.{extension}"))
