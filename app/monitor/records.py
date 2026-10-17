"""Delete only flat monitor trace records; never touch Agent execution stores."""
from __future__ import annotations

from itertools import islice
import re
import threading
import time
from uuid import uuid4

from app.agent.runtime_trace import TraceReader

_RECORD = re.compile(r'(?:meta\.json(?:\.tmp)?|events\.jsonl|(?:request|stream)-[a-z0-9_-]{1,64}\.json(?:\.tmp)?)\Z')
_DELETED = re.compile(r'\.deleted-[a-f0-9]{32}-[a-f0-9]{32}\Z')


class MonitorRecords:
    def __init__(self, reader: TraceReader):
        self.reader = reader
        self.lock = threading.RLock()

    def _files(self, directory):
        if (self.reader.root.is_symlink() or directory.is_symlink()
                or directory.resolve().parent != self.reader.root.resolve()):
            raise ValueError('trace_path_invalid')
        files = list(islice(directory.iterdir(), 257))
        if len(files) > 256 or any(not _RECORD.fullmatch(item.name) or item.is_symlink()
                                  or not item.is_file() for item in files):
            raise ValueError('trace_record_directory_invalid')
        return files

    def _purge(self, directory):
        for item in self._files(directory):
            item.unlink(missing_ok=True)
        directory.rmdir()

    def purge_pending(self):
        if not self.reader.root.exists():
            return
        with self.lock:
            for directory in islice(self.reader.root.iterdir(), 1000):
                if not _DELETED.fullmatch(directory.name):
                    continue
                try:
                    self._purge(directory)
                except (OSError, ValueError):
                    # A closed Windows handle can be retried on the next delete
                    # or viewer restart. Hidden records never enter the list.
                    continue

    def delete_traces(self, trace_ids):
        if not isinstance(trace_ids, (list, tuple)) or len(trace_ids) > 500:
            raise ValueError('trace_delete_selection_invalid')
        with self.lock:
            directories = [self.reader._directory(identifier) for identifier in dict.fromkeys(trace_ids)]
            for directory in directories:
                self._files(directory)  # Validate the complete snapshot before any deletion.
            deleted, pending = [], 0
            for directory in directories:
                target = self.reader.root / ('.deleted-' + directory.name + '-' + uuid4().hex)
                directory.rename(target)
                deleted.append(directory.name)
                try:
                    self._purge(target)
                except (OSError, ValueError):
                    pending += 1
            self.purge_pending()
            return {'deleted_turns': len(deleted), 'trace_ids': deleted, 'cleanup_pending': bool(pending),
                    'scope': 'monitor_records_only', 'execution_cancelled': False}

    def delete_conversation(self, conversation_id):
        with self.lock:
            conversation = self.reader.read_conversation(conversation_id, include_reset_history=True)
            return self.delete_traces([meta['trace_id'] for meta in
                (*conversation['turns'], *conversation.get('task_traces', ()))])

    def delete_before(self, cutoff):
        if type(cutoff) not in (float, int) or not 0 < cutoff <= time.time():
            raise ValueError('trace_delete_cutoff_invalid')
        with self.lock:
            return self.delete_traces([meta['trace_id'] for meta in self.reader.list_traces(limit=500)
                if type(meta.get('started_at')) in (int, float) and meta['started_at'] < cutoff])
