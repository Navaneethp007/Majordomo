"""Firing the briefing on wake / boot / login, via Windows Task Scheduler.

Three separate tasks rather than one, because the three events are registered
differently: logon and boot have first-class triggers, while "woke from sleep"
does not — it only exists as an event-log record (Kernel-Power id 107 in
``System``), so it needs an event trigger with an XPath query.

Registration goes through ``schtasks /XML`` rather than the flag form: the flag
form has no way to express an event trigger at all.

The XML is built with ``xml.etree`` rather than string formatting so a task name
containing an ``&`` produces a valid document instead of a corrupt one.
"""
from __future__ import annotations

import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

_NS = "http://schemas.microsoft.com/windows/2004/02/mit/task"

TASK_PREFIX = "Majordomo"

#: What the scheduled tasks actually run. A module constant rather than a local
#: inside install(), so a test can assert on it — passing this string *into*
#: build_task_xml and checking it comes back out only proves the XML builder
#: echoes its input, and would stay green if this reverted to --no-speak.
#:
#: --speak-if-needed, not --no-speak: spec §8 has the wake cycle speak, and the
#: speech gate stops that becoming noise.
BRIEF_ARGUMENTS = "-m majordomo.cli brief --speak-if-needed"

#: Kernel-Power 107 is logged when the system resumes from sleep or hibernate.
WAKE_QUERY = (
    "<QueryList><Query Id='0' Path='System'>"
    "<Select Path='System'>"
    "*[System[Provider[@Name='Microsoft-Windows-Kernel-Power'] and EventID=107]]"
    "</Select></Query></QueryList>"
)


class TriggerError(Exception):
    """Registering or removing a scheduled task failed."""


@dataclass(frozen=True)
class TaskSpec:
    name: str
    kind: str  # "logon" | "boot" | "wake"
    description: str


TASKS = [
    TaskSpec(f"{TASK_PREFIX}\\OnLogon", "logon", "Brief me when I log in"),
    TaskSpec(f"{TASK_PREFIX}\\OnBoot", "boot", "Brief me when the machine boots"),
    TaskSpec(f"{TASK_PREFIX}\\OnWake", "wake", "Brief me when the machine wakes from sleep"),
]


def _sub(parent: ET.Element, tag: str, text: str | None = None) -> ET.Element:
    element = ET.SubElement(parent, tag)
    if text is not None:
        element.text = text
    return element


def build_task_xml(spec: TaskSpec, command: str, arguments: str) -> str:
    """Build the Task Scheduler XML document for one trigger."""
    ET.register_namespace("", _NS)
    task = ET.Element(f"{{{_NS}}}Task", {"version": "1.2"})

    registration = _sub(task, "RegistrationInfo")
    _sub(registration, "Description", spec.description)
    _sub(registration, "Author", "Majordomo")

    triggers = _sub(task, "Triggers")
    if spec.kind == "logon":
        trigger = _sub(triggers, "LogonTrigger")
        _sub(trigger, "Enabled", "true")
    elif spec.kind == "boot":
        trigger = _sub(triggers, "BootTrigger")
        _sub(trigger, "Enabled", "true")
        # The desktop is not ready the instant the boot trigger fires.
        _sub(trigger, "Delay", "PT1M")
    else:
        trigger = _sub(triggers, "EventTrigger")
        _sub(trigger, "Enabled", "true")
        _sub(trigger, "Subscription", WAKE_QUERY)
        _sub(trigger, "Delay", "PT15S")

    principals = _sub(task, "Principals")
    principal = ET.SubElement(principals, "Principal", {"id": "Author"})
    _sub(principal, "LogonType", "InteractiveToken")
    _sub(principal, "RunLevel", "LeastPrivilege")

    settings = _sub(task, "Settings")
    _sub(settings, "MultipleInstancesPolicy", "IgnoreNew")
    # A briefing is worth nothing on battery-saver silence, but it also must not
    # be the reason a laptop wakes up or stays awake.
    _sub(settings, "DisallowStartIfOnBatteries", "false")
    _sub(settings, "StopIfGoingOnBatteries", "false")
    _sub(settings, "StartWhenAvailable", "false")
    _sub(settings, "RunOnlyIfNetworkAvailable", "false")
    _sub(settings, "ExecutionTimeLimit", "PT5M")
    _sub(settings, "Enabled", "true")
    idle = _sub(settings, "IdleSettings")
    _sub(idle, "StopOnIdleEnd", "false")
    _sub(idle, "RestartOnIdle", "false")

    actions = ET.SubElement(task, "Actions", {"Context": "Author"})
    action = _sub(actions, "Exec")
    _sub(action, "Command", command)
    _sub(action, "Arguments", arguments)

    return '<?xml version="1.0" encoding="UTF-16"?>\n' + ET.tostring(task, encoding="unicode")


def _run_schtasks(args: list[str]) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(
            ["schtasks", *args], capture_output=True, text=True, timeout=30
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise TriggerError(f"could not run schtasks: {exc}") from exc


def install(python: str | None = None) -> list[str]:
    """Register all three tasks. Returns the names registered."""
    if sys.platform != "win32":
        raise TriggerError("wake/boot/login triggers are Windows-only in v1")

    command = (python or sys.executable).replace("/", "\\")
    # pythonw.exe runs without flashing a console window on every wake.
    windowless = command.replace("python.exe", "pythonw.exe")
    if not Path(windowless).is_file():
        windowless = command
    arguments = BRIEF_ARGUMENTS

    registered: list[str] = []
    for spec in TASKS:
        xml = build_task_xml(spec, windowless, arguments)
        # schtasks /XML insists on reading from a file, and on UTF-16.
        with tempfile.NamedTemporaryFile(
            "w", suffix=".xml", delete=False, encoding="utf-16"
        ) as fh:
            fh.write(xml)
            temp = fh.name
        try:
            result = _run_schtasks(["/Create", "/TN", spec.name, "/XML", temp, "/F"])
            if result.returncode != 0:
                raise TriggerError(
                    f"registering {spec.name} failed: {result.stderr.strip() or result.stdout.strip()}"
                )
            registered.append(spec.name)
        finally:
            Path(temp).unlink(missing_ok=True)

    return registered


def uninstall() -> int:
    """Remove every task we registered. Returns how many were removed."""
    removed = 0
    for spec in TASKS:
        result = _run_schtasks(["/Delete", "/TN", spec.name, "/F"])
        if result.returncode == 0:
            removed += 1
    return removed
