from typing import Optional

import os

from .utils import logger

_debugger_status_ = [False]


def pycxpress_debugger(
    host: Optional[str] = None,
    port: Optional[int] = None,
    debugger: Optional[str] = None,
):
    if _debugger_status_[0] == True:
        return

    if debugger is None:
        return

    if host is None:
        host = os.environ.get("PYCXPRESS_DEBUGGER_HOST", "localhost")

    if port is None:
        port = int(os.environ.get("PYCXPRESS_DEBUGGER_PORT", "5678"))

    if debugger.lower() == "pycharm":
        try:
            import pydevd_pycharm

            pydevd_pycharm.settrace(
                host, port=port, stdoutToServer=True, stderrToServer=True, suspend=True
            )
            _debugger_status_[0] = True

        except ConnectionRefusedError:
            logger.warning(
                "Can not connect to Python debug server (maybe not started?)"
            )
            logger.warning(
                "Use PYCXPRESS_DEBUGGER_TYPE=debugpy instead as Pycharm professional edition is needed for Python debug server feature."
            )
    elif debugger.lower() == "debugpy":
        import debugpy

        _debugger_status_[0] = True

        debugpy.listen((host, port))
        logger.info(f"debugpy listen on {host}:{port}, please use VSCode to attach")
        debugpy.wait_for_client()
    else:
        logger.warning(
            f"Only PYCXPRESS_DEBUGGER_TYPE=debugpy|pycharm supported but {debugger} provided"
        )
