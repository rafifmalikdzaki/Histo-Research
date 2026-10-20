"""Cross-process file locking for the analysis output directory.

Several training processes may share one analysis directory; the lock keeps
them from writing the same metric file at once.
"""

import fcntl

__all__ = ["FileLockMixin"]


class FileLockMixin:
    """Advisory lock around the analysis output directory."""

    def _acquire_lock(self):
        """Acquire file lock to prevent concurrent access"""
        try:
            self.lock_fd = open(self.lock_file, 'w')
            fcntl.flock(self.lock_fd.fileno(), fcntl.LOCK_EX)
            return True
        except Exception:
            return False

    def _release_lock(self):
        """Release file lock"""
        try:
            if hasattr(self, 'lock_fd'):
                fcntl.flock(self.lock_fd.fileno(), fcntl.LOCK_UN)
                self.lock_fd.close()
        except Exception:
            pass
