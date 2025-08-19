'''Necessary to mock config.py for pytest. config.py is not created by default when cloning the repository.'''
# imports
import os
import sys
import types

if "protflow.config" not in sys.modules:
    cfg = types.ModuleType("protflow.config")
    cfg.__file__ = "<pytest-mock>"
    cfg.__package__ = "protflow"

    cfg.PROTFLOW_DIR = "../"
    cfg.AUXILIARY_RUNNER_SCRIPTS_DIR = os.path.join(
        os.getcwd(), "protflow", "tools", "runners_auxiliary_scripts"
    )

    # overwrite the default attribute accession:
    def _module_getattr(name):
        return ""
    cfg.__getattr__ = _module_getattr

    sys.modules["protflow.config"] = cfg
