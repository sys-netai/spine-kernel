import os
from os import path
import sys

# Current spine-kernel src directory
src_dir = path.abspath(path.join(path.dirname(__file__), os.pardir))

# Main repository src directory (go up to third_party, then up to repo root, then to src)
spine_kernel_dir = path.abspath(path.join(src_dir, os.pardir, os.pardir))
main_repo_root = path.abspath(path.join(spine_kernel_dir, os.pardir, os.pardir))
main_repo_src_dir = path.abspath(path.join(main_repo_root, "src"))

# Log directory
log_dir = path.abspath(path.join(src_dir, os.pardir, "log"))

# Add paths for imports
sys.path.append(src_dir)  # For spine-kernel modules
sys.path.append(main_repo_src_dir)  # For main repo modules