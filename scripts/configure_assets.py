#!/usr/bin/env python3
"""Connect a storybook checkout to image backends and local assets without copying them."""
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def configure(images: Path, story: Path = ROOT) -> list[Path]:
    images = images.expanduser().resolve()
    backend = images / 'sdxl_littlequeen_v1'
    if not (backend / 'scripts/run_sdxl_cpu.py').is_file():
        raise ValueError(f'Not an images checkout: {images}')
    names = ['models', '.tooling', 'outputs', 'logs', 'archives', 'transfer']
    names += [p.name for p in backend.glob('training_dataset*') if p.is_dir()]
    names += [p.name for p in backend.glob('storybook_accessories_v*') if p.is_dir()]
    links = [(backend / name, story / 'sdxl_littlequeen_v1' / name)
             for name in names if (backend / name).is_dir()]
    # Check every destination before making changes; never replace existing data.
    for source, target in links:
        if target.is_symlink() and target.resolve() == source.resolve():
            continue
        if target.exists() or target.is_symlink():
            raise ValueError(f'Existing path is not the requested link: {target}')
    for source, target in links:
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.is_symlink():
            target.symlink_to(source, target_is_directory=True)
    settings = story / '.local/images-repo'
    settings.parent.mkdir(parents=True, exist_ok=True)
    settings.write_text(str(images) + '\n')
    return [target for _, target in links]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--images-repo', type=Path)
    parser.add_argument('--workspace', type=Path)
    args = parser.parse_args()
    if args.images_repo is None:
        import sys
        sys.path.insert(0, str(ROOT))
        from media_workspace.config import load_workspace
        configured = load_workspace(args.workspace)['components'].get('images')
        if not configured:
            parser.error('Set images in the workspace config or pass --images-repo')
        args.images_repo = Path(configured['root'])
    try:
        for target in configure(args.images_repo):
            print(target)
    except ValueError as exc:
        parser.error(str(exc))


if __name__ == '__main__':
    main()
