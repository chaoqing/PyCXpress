from typing import Optional

import os

from .utils import Singleton, logger


class Debugger(metaclass=Singleton):
    def __init__(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None,
        debugger: Optional[str] = None,
        timeout: int = 10,
    ) -> None:
        if host is None:
            host = os.environ.get("PYCXPRESS_DEBUGGER_HOST", "localhost")

        if port is None:
            port = int(os.environ.get("PYCXPRESS_DEBUGGER_PORT", "5678"))

        self.host_port = (host, port)
        if debugger is None:
            debugger = os.environ.get("PYCXPRESS_DEBUGGER_TYPE", "debugpy")

        self.debugger = debugger.lower()
        self.timeout = timeout
        self.status = False

    def start_pycharm_remote(self):
        try:
            import pydevd_pycharm

            host, port = self.host_port

            logger.warning(
                f"Connectting '{host}:{port}' Pycharm remote debugging server..."
            )
            pydevd_pycharm.settrace(
                host,
                port=port,
                stdoutToServer=True,
                stderrToServer=True,
                suspend=True,
            )
            self.status = True

        except ConnectionRefusedError:
            logger.warning(
                "Can not connect to Python debug server (maybe not started?)"
            )
            logger.warning(
                "Use PYCXPRESS_DEBUGGER_TYPE=debugpy|pycharm_local instead as Pycharm professional edition is needed for Python debug server feature."
            )

    def start_pycharm_local(self):
        import sys

        if "attach_script" in sys.modules:
            logger.warning(
                "There can be only one top level package `attach_script` from Pycharm bundled `plugins/python/helpers/pydev/pydevd_attach_to_process/attach_script.py`"
            )

        import os
        from time import sleep

        timeout = self.timeout
        logger.warning(
            f"Use Pycharm `Run | Attach to Process` to process {os.getpid()} in {timeout} seconds.\n"
            "See more at [here](https://www.jetbrains.com/help/pycharm/attach-to-process.html)"
        )
        while timeout > 0:
            sleep(1)
            timeout -= 1
            if "attach_script" in sys.modules:
                self.status = True
                break

    def start_debugpy(self):
        import debugpy

        debugpy.listen(self.host_port)
        host, port = self.host_port
        logger.warning(f"debugpy listen on {host}:{port}, please use VSCode to attach")
        debugpy.wait_for_client()
        self.status = True

    def start(self):
        if self.status:
            return

        if self.debugger == "pycharm":
            self.start_pycharm_remote()
        elif self.debugger == "debugpy":
            self.start_debugpy()
        elif self.debugger == "pycharm_local":
            self.start_pycharm_local()
        else:
            logger.warning(
                f"Only PYCXPRESS_DEBUGGER_TYPE=debugpy|pycharm|pycharm_local supported but {self.debugger} provided"
            )


def pycxpress_debugger(
    host: Optional[str] = None,
    port: Optional[int] = None,
    debugger: Optional[str] = None,
):
    return Debugger(host, port, debugger).start()
