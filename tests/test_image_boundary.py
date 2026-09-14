import importlib.util
from pathlib import Path

import pytest

import image_backend

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/configure_assets.py'
SPEC = importlib.util.spec_from_file_location('configure_assets', SCRIPT)
ASSETS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ASSETS)


def test_adapter_uses_configured_backend(tmp_path, monkeypatch):
    scripts = tmp_path / 'sdxl_littlequeen_v1/scripts'
    scripts.mkdir(parents=True)
    (scripts / 'run_sdxl_cpu.py').write_text('VALUE = "fixture backend"\n')
    monkeypatch.setenv('IMAGES_REPO', str(tmp_path))
    assert image_backend.load_backend('run_sdxl_cpu').VALUE == 'fixture backend'
    with pytest.raises(FileNotFoundError, match='IMAGES_REPO'):
        image_backend.backend_source('recover_colab_artifact')


def test_asset_binding_is_repeatable_and_does_not_copy_data(tmp_path):
    images = tmp_path / 'images'
    root = images / 'sdxl_littlequeen_v1'
    (root / 'scripts').mkdir(parents=True)
    (root / 'scripts/run_sdxl_cpu.py').touch()
    (root / 'models').mkdir()
    (root / 'models/fixture').write_text('preserve')
    story = tmp_path / 'story'
    ASSETS.configure(images, story)
    ASSETS.configure(images, story)
    assert (story / 'sdxl_littlequeen_v1/models').is_symlink()
    assert (story / 'sdxl_littlequeen_v1/models/fixture').read_text() == 'preserve'


def test_asset_binding_refuses_existing_directory(tmp_path):
    images = tmp_path / 'images'
    root = images / 'sdxl_littlequeen_v1'
    (root / 'scripts').mkdir(parents=True)
    (root / 'scripts/run_sdxl_cpu.py').touch()
    (root / 'models').mkdir()
    story = tmp_path / 'story'
    (story / 'sdxl_littlequeen_v1/models').mkdir(parents=True)
    with pytest.raises(ValueError, match='Existing path'):
        ASSETS.configure(images, story)
    assert not (story / '.local/images-repo').exists()
