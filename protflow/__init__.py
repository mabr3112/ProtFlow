'''Package initialization'''
from __future__ import annotations
import os
import sys
import importlib.util
from shutil import which
from pathlib import Path
from typing import Optional
from importlib import import_module

class MissingConfigError(ImportError):
    pass

class ProtFlowConfigError(RuntimeError):
    '''Error to be raised when a variable is missing in config.py (e.g. ESM_PATH is not in there.)'''
    def __init__(self, config: object, var: str):
        config_path = getattr(config, "__file__", "not set, run protflow-init-config in your terminal!")
        message = f"""Missing parameter in config.py: {var}
        Please add this parameter and its path to your config file.
        Current config file: {config_path}
        """
        super().__init__(message)

class MissingConfigSettingError(RuntimeError):
    '''Error to be raised when a variable is not set in config.py (e.g. ESM_PATH = "")'''
    def __init__(self, config: object, var: str):
        config_path = getattr(config, "__file__", "not set, run protflow-init-config in your terminal!")
        message = f"Variable {var} not specified in config.py. Please specify path!\nYour config.py {config_path}"
        super().__init__(message)

def _expand(v: str) -> str:
    # Expand ~ and $VARS, keep as string for which()
    return Path(v).expanduser().as_posix().replace("~", str(Path.home()))

def _xdg_config_dir() -> Path:
    base = os.environ.get("XDG_CONFIG_HOME") or os.path.join(Path.home(), ".config")
    return Path(base) / "protflow"

def _xdg_config_path() -> Path:
    return _xdg_config_dir / "config.py"

def _saved_config_pointer_path() -> Path:
    return _xdg_config_dir() / "config.path"

def _read_saved_config_path() -> Optional[str]:
    '''Find path to pointer (_xdg_config_dir/config.path). 
    Then check if it is file and if it contains text.
    Only if it does, return the pointer path.
    Otherwise return None.
    '''
    pointer_path = _saved_config_pointer_path()
    if not pointer_path.is_file():
        return None
    try:
        text = pointer_path.read_text().strip()
    except Exception: # pylint: disable=W0718
        return None
    if not text:
        return None
    return str(Path(text).expanduser())

def _load_module_from_file(module_spec: str, file_path: str) -> Optional[object]:
    # create a specification from the file location
    spec = importlib.util.spec_from_file_location(module_spec, file_path)

    # set up all module attributes and stuff before loading
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader

    # ensure the module is available afterwards globally for importing
    sys.modules["protflow.config"] = module

    # load module (execute file) # Note: Potential security issue, because user-specified code (config.py) is executed!
    spec.loader.exec_module(module)
    return module

def _try_load_config_module() -> Optional[object]:
    '''
    Tries to load config.py in the following order:
        0: custom set path by >$ protflow-set-config
            custom path
        1: environment variable:
            PROTFLOW_CONFIG 
        2: default config destination:
            /user/home/.config/protflow/config.py
        3: at the package:
            /path/to/protflow/config.py
    '''
    # 0: try loading custom path if it was set:
    config_path = _read_saved_config_path()
    if config_path and Path(config_path).is_file():
        return _load_module_from_file("protflow.config", config_path)

    # 1: try load from environment variable (overwrites default config destination)
    config_path = os.getenv("PROTFLOW_CONFIG")
    if config_path and Path(config_path).is_file():
        return _load_module_from_file("protflow.config", config_path)

    # 2: try loading from default path /home/<user>/.config/protflow/config.py
    config_path = _xdg_config_path()
    if config_path.is_file():
        return _load_module_from_file("protflow.config", str(config_path))

    # 3: try loading from package root
    try:
        from . import config as module # pylint: disable=E0611
        return module
    except Exception: # pylint: disable=W0718
        return None

__CONFIG = _try_load_config_module()

def require_config() -> object:
    """Default function to be called in runners to require a set-up config.py file. 
    This function imports and returns protflow.config"""
    # return config module if it is set up
    if __CONFIG is not None:
        return __CONFIG

    # if config was not set up yet, print instructive message for user to set up config.
    package_root = Path(__file__).resolve().parent
    template = package_root / "config_template.py"
    msg = f"""
ProtFlow configuration missing (config.py).

Run one of:
  protflow-init-config
  protflow-init-config --dest /path/to/config.py
  protflow-set-config /absolute/path/to/config.py   # pins a specific file for future runs

Or set an explicit path (and make sure the PROTFLOW_CONFIG environment variable is always set when running protflow):
  export PROTFLOW_CONFIG=/absolute/path/to/config.py

Search order:
  0) $XDG_CONFIG_HOME/protflow/config.path (saved by protflow-set-config)
  1) $PROTFLOW_CONFIG
  2) $XDG_CONFIG_HOME/protflow/config.py (or ~/.config/protflow/config.py)
  3) bundled protflow/config.py

Template:
  {template}

Docs: https://github.com/mabr3112/ProtFlow
""".strip()
    raise MissingConfigError(msg)

def load_config_path(config: object, path_var: str, is_pre_cmd: bool = False) -> Optional[str]:
    '''
    Loads a variable from config.py
    If the variable is not set, it returns an error message to set the variable.
    '''
    try:
        var = getattr(config, path_var)
    except AttributeError as exc:
        raise ProtFlowConfigError(config, path_var) from exc

    # if the loaded config setting is a pre_cmd, return without checking
    if is_pre_cmd:
        return var

    # variable must be set
    if not var:
        raise MissingConfigSettingError(config, path_var)

    # in case we have an executable, return
    var = var if ("/" in var or "\\" in var) else which(var)

    # check if file exists and return
    out_path = Path(_expand(var)).resolve()
    if out_path.exists():
        return out_path
    raise FileNotFoundError(out_path)

def get_config() -> object:
    return __CONFIG

# keep top-level light; lazy-load heavy subpackages
__all__ = ["require_config", "get_config"]

# define packages that should be accessible by default here:
DEFAULT_PACKAGES = [
    "poses", "jobstarters", "runners", "residues", 
    "tools", "metrics", "utils"
]

def __getattr__(name: str):
    if name in DEFAULT_PACKAGES:
        mod = import_module(f".{name}", __name__)
        globals()[name] = mod
        return mod
    raise AttributeError(name)
