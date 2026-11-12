"""Standard output-capture helper kept for harness compatibility."""

from __future__ import annotations

import sys
from pathlib import Path


class _Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, text):
        for stream in self.streams:
            stream.write(text)
        return len(text)

    def flush(self):
        for stream in self.streams:
            stream.flush()


class OutputCapture:
    """Context manager that mirrors stdout/stderr into a log file."""

    def __init__(self, log_path):
        self.log_path = Path(log_path)
        self._file = None
        self._saved = None

    def __enter__(self):
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(self.log_path, "a", encoding="utf-8")
        self._saved = (sys.stdout, sys.stderr)
        sys.stdout = _Tee(self._saved[0], self._file)
        sys.stderr = _Tee(self._saved[1], self._file)
        return self

    def __exit__(self, *exc):
        sys.stdout, sys.stderr = self._saved
        self._file.close()
        return False
