#!/usr/bin/env python3

# Load centralized workstation defaults; explicit environment/CLI values win.
import sys as _workspace_sys
from pathlib import Path as _WorkspacePath
for _workspace_root in _WorkspacePath(__file__).resolve().parents:
    if (_workspace_root / "media_workspace").is_dir():
        _workspace_sys.path.insert(0, str(_workspace_root))
        break
from media_workspace.config import apply_environment as _apply_workspace
_apply_workspace()

"""Run the selected story workflow using this process's Python environment."""
import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['colab', 'q8'], default='colab')
    parser.add_argument('--story', type=Path, required=True)
    parser.add_argument('--run-dir', type=Path, required=True)
    args = parser.parse_args()
    script = ROOT / 'sdxl_littlequeen_v1/storybook_mvp_v1' / (
        'run_storybook_q8.sh' if args.mode == 'q8' else 'run_storybook.sh')
    if not args.story.is_file():
        parser.error(f'Story does not exist: {args.story}')
    env = {**os.environ, 'PATH': str(Path(sys.executable).parent) + os.pathsep + os.environ.get('PATH', '')}
    return subprocess.run(['bash', str(script), str(args.story.resolve()), str(args.run_dir.resolve())],
                          cwd=ROOT, env=env, check=False).returncode


if __name__ == '__main__':
    raise SystemExit(main())
