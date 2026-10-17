"""Metadata serialization must queue safely rather than fail ordinary siblings."""
from concurrent.futures import ThreadPoolExecutor
from threading import BoundedSemaphore, Event
import time

import pytest

from app.providers.x_crawler_client import XCrawlerClient
from test_x_crawler_client import client_with, PID, URL


def test_three_metadata_calls_queue_and_preserve_single_worker(monkeypatch):
    slots=BoundedSemaphore(1);entered=Event();release=Event();waiting=Event();calls=[];checks=[]
    monkeypatch.setattr(XCrawlerClient,'_slots',slots)
    def observe(*args):
        calls.append(True)
        if len(calls)==1:entered.set();assert release.wait(3)
    def check():
        checks.append(True)
        if len(checks)>=4:waiting.set()
    with ThreadPoolExecutor(3) as pool:
        first=pool.submit(client_with(observe=observe).inspect,URL)
        assert entered.wait(2)
        others=[pool.submit(client_with(observe=observe).inspect,URL,check=check) for _ in range(2)]
        try:
            assert waiting.wait(2)  # Both waiters have polled and rechecked authority.
            assert len(calls)==1
        finally:release.set()
        assert [f.result(timeout=3)['post_id'] for f in [first,*others]]==[PID]*3


def test_waiting_metadata_stops_on_revoked_source_without_spawning_worker(monkeypatch):
    slots=BoundedSemaphore(1);slots.acquire();calls=[];checks=[]
    monkeypatch.setattr(XCrawlerClient,'_slots',slots)
    def check():
        checks.append(True)
        if len(checks)==2:raise RuntimeError('synthetic_source_revoked')
    try:
        with pytest.raises(RuntimeError,match='source_revoked'):
            client_with(observe=lambda *a:calls.append(True)).inspect(URL,check=check)
    finally:slots.release()
    assert not calls and slots.acquire(blocking=False)
    slots.release()
