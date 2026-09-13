"""Tests de portabilidad del ``config.yaml`` del checkpoint oficial.

El checkpoint de TRIBE v2 se serializó en Linux con ``yaml.UnsafeLoader`` e incluye
objetos ``pathlib.PosixPath`` que Windows no puede instanciar
(``NotImplementedError: cannot instantiate 'PosixPath' on your system``).
"""

from pathlib import Path

import pytest
import yaml

from core.tribe_model import portable_config_text

# Fragmento literal del config.yaml oficial de facebook/tribev2.
REAL_FRAGMENT = """data:
    video:
        chunkvideos:
            infra:
                folder: !!python/object/apply:pathlib.PosixPath
                - /
                - checkpoint
                - sdascoli
                - cache
                - tribe_release
                cache_type: ValidatedParquet
                mode: cached
    audio_feature:
        infra:
            folder: /checkpoint/sdascoli/cache/tribe_release
"""

LOCAL_ROOT = "cache/meta_checkpoint_cache"


def test_documents_the_windows_failure():
    """Confirmación de que el YAML original es incargable en esta plataforma."""
    with pytest.raises(Exception):
        yaml.load(REAL_FRAGMENT, Loader=yaml.UnsafeLoader)


def test_rewritten_config_loads():
    """Tras la transformación, el YAML carga sin excepción."""
    out = yaml.load(portable_config_text(REAL_FRAGMENT, LOCAL_ROOT), Loader=yaml.UnsafeLoader)
    assert out is not None
    assert "data" in out


def test_posixpath_becomes_a_local_string():
    out = yaml.load(portable_config_text(REAL_FRAGMENT, LOCAL_ROOT), Loader=yaml.UnsafeLoader)
    folder = out["data"]["video"]["chunkvideos"]["infra"]["folder"]

    assert isinstance(folder, str)
    assert folder.startswith(LOCAL_ROOT)
    assert folder.endswith("checkpoint/sdascoli/cache/tribe_release")
    # Separa correctamente los componentes en lugar de concatenarlos.
    assert "checkpointsdascoli" not in folder


def test_absolute_plain_paths_are_localized():
    out = yaml.load(portable_config_text(REAL_FRAGMENT, LOCAL_ROOT), Loader=yaml.UnsafeLoader)
    folder = out["data"]["audio_feature"]["infra"]["folder"]

    assert isinstance(folder, str)
    assert folder.startswith(LOCAL_ROOT)
    # No debe quedar una ruta absoluta en la raíz de la unidad.
    assert not Path(folder).is_absolute() or LOCAL_ROOT.split("/")[0] in folder


def test_transform_is_idempotent():
    once = portable_config_text(REAL_FRAGMENT, LOCAL_ROOT)
    twice = portable_config_text(once, LOCAL_ROOT)
    assert once == twice


def test_unrelated_yaml_keys_are_preserved():
    out = yaml.load(portable_config_text(REAL_FRAGMENT, LOCAL_ROOT), Loader=yaml.UnsafeLoader)
    assert out["data"]["video"]["chunkvideos"]["infra"]["cache_type"] == "ValidatedParquet"
    assert out["data"]["video"]["chunkvideos"]["infra"]["mode"] == "cached"


def test_real_downloaded_config_is_loadable_if_present():
    """Si el checkpoint ya está descargado, se comprueba sobre el archivo real."""
    config = Path("cache/checkpoint/facebook__tribev2/config.yaml")
    if not config.exists():
        pytest.skip("checkpoint no descargado")

    text = config.read_text(encoding="utf-8")
    if "!!python/object/apply" not in text:
        pytest.skip("config ya transformado")

    assert "!!python/object/apply" in text
    out = yaml.load(portable_config_text(text, LOCAL_ROOT), Loader=yaml.UnsafeLoader)
    assert out["average_subjects"] is True or "data" in out
