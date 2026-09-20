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

import getpass
import os
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


#: Two triggers, not three.
#:
#: There is no OnBoot. A BootTrigger runs before anyone logs in, so Windows
#: treats it as a machine-level task and refuses to register it without
#: elevation — and on a personal machine it buys nothing anyway, because a boot
#: is always followed by a logon. All it ever contributed was a second briefing
#: a minute after the first, and a UAC prompt during install.
#:
#: Both of these register as the current user with no elevation.
TASKS = [
    TaskSpec(f"{TASK_PREFIX}\\OnLogon", "logon", "Brief me when I log in (covers boot too)"),
    TaskSpec(f"{TASK_PREFIX}\\OnWake", "wake", "Brief me when the machine wakes from sleep"),
]

#: Tasks earlier versions registered and this one no longer creates.
#:
#: Removing OnBoot from TASKS alone would have made it *unremovable*: uninstall
#: only sweeps what TASKS lists, so anyone who had already installed would keep
#: a live Majordomo\OnBoot firing a second briefing a minute after logon — the
#: exact duplicate the removal was meant to fix, now beyond the reach of both
#: install and uninstall. So both sweep this list too.
LEGACY_TASK_NAMES = [f"{TASK_PREFIX}\\OnBoot"]


def current_user() -> str:
    """DOMAIN\\user for the task principal.

    Without a UserId, a LogonTrigger applies to *every* user on the machine,
    which requires administrator rights — that is the whole reason
    `mj install-trigger` used to fail with "Access is denied."
    """
    domain = os.environ.get("USERDOMAIN", "")
    user = getpass.getuser()
    return f"{domain}\\{user}" if domain else user


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

    user = current_user()

    triggers = _sub(task, "Triggers")
    if spec.kind == "logon":
        trigger = _sub(triggers, "LogonTrigger")
        _sub(trigger, "Enabled", "true")
        # Scope to this user. Without it the trigger covers every account on
        # the machine, which needs administrator rights to register.
        _sub(trigger, "UserId", user)
        # The desktop is not usable the instant you log in.
        _sub(trigger, "Delay", "PT30S")
    else:
        trigger = _sub(triggers, "EventTrigger")
        _sub(trigger, "Enabled", "true")
        _sub(trigger, "Subscription", WAKE_QUERY)
        _sub(trigger, "Delay", "PT15S")

    principals = _sub(task, "Principals")
    principal = ET.SubElement(principals, "Principal", {"id": "Author"})
    _sub(principal, "UserId", user)
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

    # Deliberately no PYTHONUTF8 wrapper here. Forcing it would mean running via
    # `cmd /c`, which flashes a console window on every wake — exactly what
    # pythonw.exe is chosen to avoid. It is also unnecessary: under pythonw
    # there is no console to encode *to*, and `safe_print` handles the
    # interactive cp1252 case where the problem actually exists.

    return '<?xml version="1.0" encoding="UTF-16"?>\n' + ET.tostring(task, encoding="unicode")


def _run_schtasks(args: list[str]) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(
            ["schtasks", *args], capture_output=True, text=True, timeout=30
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise TriggerError(f"could not run schtasks: {exc}") from exc


def install(python: str | None = None) -> tuple[list[str], list[str]]:
    """Register all three tasks.

    Returns ``(registered, warnings)``. A task that would not register is a
    warning rather than a failure — refusing everything because one of three
    was rejected leaves you worse off than the partial install does.
    """
    if sys.platform != "win32":
        raise TriggerError("wake/boot/login triggers are Windows-only in v1")

    command = (python or sys.executable).replace("/", "\\")
    # pythonw.exe runs without flashing a console window on every wake.
    windowless = command.replace("python.exe", "pythonw.exe")
    if not Path(windowless).is_file():
        windowless = command
    arguments = BRIEF_ARGUMENTS

    registered: list[str] = []
    failed: list[tuple[str, str]] = []

    # Retire anything an older version left behind before adding the new set.
    for name in LEGACY_TASK_NAMES:
        _run_schtasks(["/Delete", "/TN", name, "/F"])

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
            if result.returncode == 0:
                registered.append(spec.name)
            else:
                failed.append((spec.name, (result.stderr or result.stdout).strip()))
        finally:
            Path(temp).unlink(missing_ok=True)

    # Report per task rather than aborting on the first failure. Raising after
    # one refusal meant a single unregisterable trigger left you with *nothing*
    # installed, when the others were perfectly fine.
    if failed and not registered:
        detail = "; ".join(f"{name}: {err}" for name, err in failed)
        raise TriggerError(detail)

    # Returned rather than printed. Deciding *what* happened belongs here;
    # deciding how to say it belongs to whoever called — the same split
    # `scaffold.create` and `activity.fetch` already make, and the reason this
    # module needs no console.
    warnings = [f"could not register {name}: {err}" for name, err in failed]
    return registered, warnings


def uninstall() -> int:
    """Remove every task we registered. Returns how many were removed."""
    removed = 0
    # Legacy names included, or a task this version stopped creating could never
    # be uninstalled by the version that stopped creating it.
    for name in [spec.name for spec in TASKS] + LEGACY_TASK_NAMES:
        result = _run_schtasks(["/Delete", "/TN", name, "/F"])
        if result.returncode == 0:
            removed += 1
    return removed
