"""Utility functions related to logging."""
import logging
from pathlib import Path
from datetime import datetime


class StreamToLogger(object):
    """Fake file-like stream object that redirects writes to a logger instance.

    Taken from: https://stackoverflow.com/a/39215961

    """

    def __init__(self, logger, level):
        """Initialize logger.

        Parameters
        ----------
        logger : logging.Logger
            Logger instance
        level : logging.LEVEL
            Logging level

        """
        self.logger = logger
        self.level = level
        self.linebuf = ""

    def write(self, buf):
        """Write to logger.

        Parameters
        ----------
        buf : str-like
            Buffer that is written to logger

        """
        for line in buf.rstrip().splitlines():
            self.logger.log(self.level, line.rstrip())

    def flush(self):
        """Flush buffer."""
        pass
