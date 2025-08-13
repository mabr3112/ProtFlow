'''Commandline scripts to setup and configure ProtFlow.'''
# imports
from __future__ import annotations
import os
import sys
import shutil
import argparse
from pathlib import Path

# declare globals
TEMPLATE_NAME = "config_template.py"
CONFIG_BASENAME = "config.py"

def _xdg_config_home() -> Path:
    # Linux/XDG default
    base = os.environ.get("XDG_CONFIG_HOME") or os.path.join(Path.home(), ".config")
    return Path(base) / "protflow"

def _pointer_path() -> Path:
    return _xdg_config_home() / "config.path"

def _package_root() -> Path:
    # should return /path/to/protflow
    return Path(__file__).resolve().parent

def _template_path() -> Path:
    return _package_root() / TEMPLATE_NAME

def _default_user_config() -> Path:
    return _xdg_config_home() / CONFIG_BASENAME

def init_config(argv: list[str] | None = None) -> None:
    '''
    Create a protflow config file from the template (config_template.py).

    Default location (Linux): $XDG_CONFIG_HOME/protflow/config.py
        -> This usually defaults to /home/<user>/.config/protflow/config.py
    You can pass a custom path: `protflow-init-config /path/to/config.py`
    Use --force to overwrite.
    '''
    default_path = str(_default_user_config())

    # setup cli args
    parser = argparse.ArgumentParser(
        prog="protflow-init-config",
        description="Create a protflow config.py from the shipped template."
    )
    parser.add_argument("--dest", type=str, help=f"Destination file path (default: {default_path})")
    parser.add_argument("-f", "--force", action="store_true", help="overwrite if the destination file already exists")
    args = parser.parse_args(argv)

    # define template
    template = _template_path()
    if not template.is_file():
        print(f"Template not found: {template}", file=sys.stderr)
        sys.exit(2)

    # create target dir
    target = Path(args.dest) if args.dest else _default_user_config()
    target.parent.mkdir(parents=True, exist_ok=True)

    # skip if config.py is already there
    if target.exists() and not args.force:
        print(f"Config already exists: {target}\n(use --force to overwrite)", file=sys.stderr)
        sys.exit(0)

    shutil.copyfile(template, target)
    print(f"Created config: {target}")
    print("Edit this file to set tool paths for ProtFlow!")

def set_config(argv: list[str] | None = None) -> None:
    '''Set config.py file destination.'''
    parser = argparse.ArgumentParser(
        prog="protflow-set-config",
        description="Set or inspect the default config.py used by ProtFlow"
    )
    parser.add_argument("path", nargs="?", help="Path to an existing config.py (absolute or relative).")
    parser.add_argument("--show", action="store_true", help="Print the saved default config path and exit.")
    parser.add_argument("--unset", action="store_true", help="Remove the saved default path.")
    args = parser.parse_args(argv)

    # define path to pointer first
    pointer = _pointer_path()

    # functionality: just show me what config is used
    if args.show:
        if pointer.is_file():
            print(pointer.read_text().strip())
            return
        print("No saved config path.", file=sys.stderr)
        sys.exit(1)

    # functionality: revert custom set path to config file
    if args.unset:
        if pointer.exists():
            pointer.unlink()
            print("Cleared saved config path.")
        else:
            print("No saved config path to clear.")
        return

    # sanity
    if not args.path:
        parser.error("Provide PATH to config.py or use --show/--unset")

    # now set new target
    target = Path(args.path).expanduser()
    if not target.is_absolute():
        target = target.resolve()

    if not target.is_file():
        print(f"Not found: {target}\nWrong path specified?", file=sys.stderr)
        sys.exit(2)
    pointer.parent.mkdir(parents=True, exist_ok=True)
    tmp = pointer.with_suffix(".tmp")
    tmp.replace(pointer)
    print(f"Saved default config path: {target}")
    print(f"(Pointer file: {pointer})")

def check_config() -> None:
    """Print which config.py file ProtFlow is using."""
    from . import get_config # pylint: disable=C0415
    cfg = get_config()

    # if there is no cfg, print error
    if cfg is None:
        print(
            "No config loaded. ProtFlow will look in:\n"
            "  0) saved path from protflow-set-config ($XDG_CONFIG_HOME/protflow/config.path)\n"
            "  1) $PROTFLOW_CONFIG\n"
            "  2) $XDG_CONFIG_HOME/protflow/config.py (or ~/.config/protflow/config.py)\n"
            "  3) site-packages/protflow/config.py\n"
            "Run: protflow-init-config", 
            file=sys.stderr
        )
        # non-zero to help CI scripts detect missing config -> This means pytest workflows need to setup a config file by running ```protflow-init-config````
        sys.exit(1)

    # if config exists, show its path:
    cfg_path = getattr(cfg, "__file__", "<in-memory>")
    print(f"Using config: {cfg_path}")
