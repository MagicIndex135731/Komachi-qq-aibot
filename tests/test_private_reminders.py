from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest

from app.adapters.sender import OutboundPrivateMessage
from app.private_reminders import PrivateReminder, PrivateReminderScheduler, load_private_reminders


class FakeSender:
    def __init__(self) -> None:
        self.private_sent: list[OutboundPrivateMessage] = []

    async def send_private_text(self, outbound: OutboundPrivateMessage) -> None:
        self.private_sent.append(outbound)


def test_load_private_reminders_reads_yaml(tmp_path) -> None:
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    (config_dir / "private_reminders.yaml").write_text(
        "\n".join(
            [
                "reminders:",
                "  - id: sample-wake-up-reminder",
                "    user_id: 20002",
                "    text: 熟人A，八点啦。",
                "    run_at: '2026-05-11T08:00:00+08:00'",
                "    catch_up_if_missed: true",
            ]
        ),
        encoding="utf-8",
    )

    reminders = load_private_reminders(config_dir=config_dir)

    assert reminders == [
        PrivateReminder(
            reminder_id="sample-wake-up-reminder",
            user_id=20002,
            text="熟人A，八点啦。",
            run_at=datetime.fromisoformat("2026-05-11T08:00:00+08:00"),
            catch_up_if_missed=True,
        )
    ]


@pytest.mark.asyncio
async def test_private_reminder_scheduler_catches_up_missed_one_time_reminder(tmp_path) -> None:
    sender = FakeSender()
    scheduler = PrivateReminderScheduler(
        sender=sender,
        data_dir=tmp_path / "data",
        reminders=[
            PrivateReminder(
                reminder_id="sample-wake-up-reminder",
                user_id=20002,
                text="熟人A，八点啦，还不起床吗？",
                run_at=datetime.fromisoformat("2026-05-11T08:00:00+08:00"),
                catch_up_if_missed=True,
            )
        ],
        allowed_user_ids={987654321, 20002},
        now_provider=lambda: datetime.fromisoformat("2026-05-11T08:05:00+08:00"),
        poll_interval_seconds=0.01,
    )

    await scheduler.start()
    await scheduler.stop()

    assert sender.private_sent == [
        OutboundPrivateMessage(user_id=20002, text="熟人A，八点啦，还不起床吗？")
    ]

    second_sender = FakeSender()
    second_scheduler = PrivateReminderScheduler(
        sender=second_sender,
        data_dir=tmp_path / "data",
        reminders=[
            PrivateReminder(
                reminder_id="sample-wake-up-reminder",
                user_id=20002,
                text="熟人A，八点啦，还不起床吗？",
                run_at=datetime.fromisoformat("2026-05-11T08:00:00+08:00"),
                catch_up_if_missed=True,
            )
        ],
        allowed_user_ids={987654321, 20002},
        now_provider=lambda: datetime.fromisoformat("2026-05-11T08:06:00+08:00"),
        poll_interval_seconds=0.01,
    )

    await second_scheduler.start()
    await second_scheduler.stop()

    assert second_sender.private_sent == []
