from __future__ import annotations

from contextlib import contextmanager, redirect_stderr, redirect_stdout
from datetime import datetime
from pathlib import Path
import sys
from typing import TextIO


class TimestampedFileStream:
    """Write stream text to a file with timestamps at line boundaries."""

    def __init__(self, file: TextIO, stream_name: str):
        self.file = file
        self.stream_name = stream_name
        self.encoding = getattr(file, "encoding", "utf-8")
        self._line_start = True

    def write(self, data: str) -> int:
        for chunk in data.splitlines(keepends=True):
            if self._line_start and chunk:
                timestamp = datetime.now().isoformat(timespec="seconds")
                self.file.write(f"{timestamp} | {self.stream_name:<6} | ")
            self.file.write(chunk)
            self._line_start = chunk.endswith("\n")
        self.file.flush()
        return len(data)

    def flush(self) -> None:
        self.file.flush()

    def isatty(self) -> bool:
        return False


class TeeStream:
    """Mirror writes to multiple stream-like objects."""

    def __init__(self, *streams):
        self.streams = streams
        self.encoding = getattr(streams[0], "encoding", "utf-8") if streams else "utf-8"

    def write(self, data: str) -> int:
        for stream in self.streams:
            stream.write(data)
            stream.flush()
        return len(data)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()

    def isatty(self) -> bool:
        return any(getattr(stream, "isatty", lambda: False)() for stream in self.streams)


@contextmanager
def tee_std_streams(log_file: Path | None, *, run_label: str):
    """Mirror stdout and stderr to a timestamped log file."""

    if log_file is None:
        yield
        return

    log_file.parent.mkdir(parents=True, exist_ok=True)
    started_at = datetime.now().isoformat(timespec="seconds")

    with log_file.open("a", encoding="utf-8") as file:
        file.write(f"\n=== {run_label} started {started_at} ===\n")
        file.flush()

        stdout = TeeStream(sys.stdout, TimestampedFileStream(file, "stdout"))
        stderr = TeeStream(sys.stderr, TimestampedFileStream(file, "stderr"))

        with redirect_stdout(stdout), redirect_stderr(stderr):
            print(f"log file: {log_file}")
            try:
                yield
            finally:
                finished_at = datetime.now().isoformat(timespec="seconds")
                print(f"{run_label} finished {finished_at}")
