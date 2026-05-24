"""Container-isolated proposer launcher (shared by all 4 sibling skills).

The previous "physical mv" isolation in meta_harness_components_*.py
moved sibling directories out of the working tree during proposer execution.
That hid them from the proposer but also hid them from any other process
using the same working tree — concretely, gaia's iter6 proposer mv'd
`meta_harness/logs_components_toolathlon/` away while toolathlon's main
loop was reading its own iter1_train_summary.jsonl, causing a
FileNotFoundError that killed the toolathlon evolution.

This module replaces that with a **whitelist bind-mount**: each skill
declares which host paths a proposer is allowed to see, and the proposer
runs inside `docker-claude:latest`. Sibling skill paths (logs_components_*,
agent_*/, mh_iter*/, etc.) are not mounted, so the container's filesystem
view simply does not contain them. The sibling skill's main loop, running
in the host filesystem namespace, is unaffected.

Container layout
----------------
- The container's cwd is the host's $ROOT (so prompt-relative paths still
  resolve and the proposer's writes land in the right host file).
- The host's $ROOT is NOT bind-mounted as a single source; each whitelisted
  subpath is bind-mounted at its own host==container path. Anything not
  whitelisted simply does not exist inside the container.
- ~/.claude and ~/.claude.json are bind-mounted RO (subscription auth).
- --network host so the proposer can reach the Anthropic API.
- --user $(id -u):$(id -g) so files written by the proposer end up
  owned by the host user.
"""
from __future__ import annotations

import os
from pathlib import Path

DOCKER_IMAGE = os.environ.get("MH_PROPOSER_DOCKER_IMAGE", "docker-claude:latest")
HOST_CLAUDE_DIR = Path(
    os.environ.get("MH_PROPOSER_HOST_CLAUDE_DIR", str(Path.home() / ".claude"))
)
HOST_CLAUDE_JSON = Path(
    os.environ.get("MH_PROPOSER_HOST_CLAUDE_JSON", str(Path.home() / ".claude.json"))
)
CONTAINER_HOME = os.environ.get("MH_PROPOSER_CONTAINER_HOME", "/home/yuhan")

# ---------------------------------------------------------------------------
# Mount manifests.  Paths are RELATIVE to the project ROOT.
# Format: list[tuple[relpath, mode]]  mode = "ro" | "rw".
# A path that does not exist on the host is silently skipped (so the same
# manifest can name optional paths like test_task_ids.txt).
# ---------------------------------------------------------------------------

# Always-needed paths shared by every skill.
_COMMON_MOUNTS: list[tuple[str, str]] = [
    # Read-only project glue the proposer reads but must not modify.
    ("bench", "ro"),
    ("meta_harness/.claude", "ro"),
    ("meta_harness/.empty_plugins", "ro"),
    ("meta_harness/claude_wrapper.py", "ro"),
    ("meta_harness/workflows", "rw"),  # patch yaml lives here
    ("agent/base.py", "ro"),
    ("agent/llm.py", "ro"),
    ("agent/events.py", "ro"),
    ("agent/v0", "ro"),
    ("agent/component_runtime", "ro"),
    ("run_benchmark.py", "ro"),
    # Trace dump root (RO).  Each skill's traces live behind a filename
    # prefix (gaia__*, tau2bench__*, toolathlon__*) — the skill's SKILL.md
    # tells the proposer which prefix to glob.  Cross-skill leakage here
    # is a known soft constraint; mount-time enforcement would require
    # restructuring traces/ into per-skill subdirs (separate refactor).
    ("traces", "ro"),
    (".component-state", "ro"),
]

# Per-skill overlay.  Order: appended AFTER _COMMON_MOUNTS so a
# skill-specific path can override mode (e.g. flip a `traces` subdir to RW).
_SKILL_MOUNTS: dict[str, list[tuple[str, str]]] = {
    "component-harness-gaia": [
        ("agent/components", "rw"),  # gaia's new component + .bak land here
        ("meta_harness/logs_components_gaia", "rw"),
        ("meta_harness/train_task_ids.txt", "ro"),
        ("meta_harness/test_task_ids.txt", "ro"),
    ],
    "component-harness-toolathlon": [
        ("meta_harness/logs_components_toolathlon", "rw"),
        ("meta_harness/toolathlon_train_task_ids.txt", "ro"),
        ("meta_harness/toolathlon_test_task_ids.txt", "ro"),
        ("meta_harness/toolathlon_all_runnable_task_ids.txt", "ro"),
        # Toolathlon has its own runtime + components dir (separate from
        # agent/components/, which is shared with gaia/tau2).
        ("agent_toolathlon/component_runtime", "ro"),
        ("agent_toolathlon/components", "rw"),
        # Per-task dumps the proposer reads via frontier_val.dump_dir paths.
        ("Toolathlon-runs", "ro"),
        (".component-state-toolathlon", "ro"),
    ],
    "component-harness-tau2": [
        ("meta_harness/logs_tau2_components", "rw"),
        ("agent_tau2/component_runtime", "ro"),
        ("agent_tau2/components", "rw"),
        # tau2 simulation dumps live under tau2-runs/.
        ("tau2-runs", "ro"),
    ],
}

# sopbench is per-domain (one process per domain); its skill manifest is
# resolved at call time from the DOMAIN string (see build_docker_cmd).
def _sopbench_mounts(domain: str) -> list[tuple[str, str]]:
    return [
        (f"meta_harness/logs_components_sopbench_{domain}", "rw"),
        (f"meta_harness/workflows/sopbench_{domain}.yaml", "rw"),
        (f"agent/components_sopbench_{domain}", "rw"),
        # Domain-specific trace dump root.
        (f"sopbench-runs/{domain}", "ro"),
    ]


# Anthropic / Claude Code env vars the proposer needs from the host.
# We do not pass arbitrary env vars by default — only this allowlist.
_ENV_PASSTHROUGH_PREFIXES = (
    "ANTHROPIC_",
    "CLAUDE_CODE_",
)
_ENV_PASSTHROUGH_EXACT = (
    "HTTPS_PROXY",
    "HTTP_PROXY",
    "NO_PROXY",
)


def _mount_arg(host_path: Path, mode: str) -> str:
    """Build the `--mount type=bind,...` argument string."""
    parts = [
        "type=bind",
        f"src={host_path}",
        f"dst={host_path}",  # host == container path so prompt absolutes still resolve
    ]
    if mode == "ro":
        parts.append("readonly")
    return ",".join(parts)


def build_docker_cmd(
    skill_name: str,
    root: Path,
    inner_claude_argv: list[str],
    *,
    container_name: str | None = None,
    domain: str | None = None,
    extra_mounts: list[tuple[str, str]] | None = None,
    extra_env: dict[str, str] | None = None,
) -> list[str]:
    """Wrap a `claude ...` argv list into a `docker run ...` invocation.

    Parameters
    ----------
    skill_name : str
        One of "component-harness-{gaia,toolathlon,tau2,sopbench}".  Used to
        look up the skill-specific mount manifest.
    root : Path
        Project ROOT (typically `/data/home/yuhan/robagent`).  All mount
        manifest paths are resolved against this.
    inner_claude_argv : list[str]
        The full claude argv, INCLUDING the leading "claude" element.
        The first element is replaced by the container's claude binary
        via --entrypoint, so callers can keep building commands the same
        way they would for a host-side claude invocation.
    container_name : str | None
        Optional --name for `docker ps` visibility.
    domain : str | None
        For sopbench: the domain slug (added to the mount manifest).
    extra_mounts : list[tuple[str, str]] | None
        Caller-supplied additional (relpath, mode) tuples.
    extra_env : dict[str, str] | None
        Extra env vars to set in the container.

    Returns
    -------
    list[str]
        Argv suitable for subprocess.Popen.  Stdout/stderr stream as usual.
    """
    if inner_claude_argv and inner_claude_argv[0] == "claude":
        claude_args = inner_claude_argv[1:]
    else:
        claude_args = list(inner_claude_argv)

    uid = os.geteuid()
    gid = os.getegid()

    cmd: list[str] = [
        "docker", "run", "--rm",
        "--user", f"{uid}:{gid}",
        "--network", "host",
        "--workdir", str(root),
        "-e", f"HOME={CONTAINER_HOME}",
    ]
    if container_name:
        cmd += ["--name", container_name]

    # Credentials.
    if HOST_CLAUDE_DIR.exists():
        cmd += ["--mount",
                f"type=bind,src={HOST_CLAUDE_DIR},dst={CONTAINER_HOME}/.claude,readonly"]
    if HOST_CLAUDE_JSON.exists():
        cmd += ["--mount",
                f"type=bind,src={HOST_CLAUDE_JSON},dst={CONTAINER_HOME}/.claude.json,readonly"]

    # Workspace mounts.
    mounts: list[tuple[str, str]] = list(_COMMON_MOUNTS)
    mounts += _SKILL_MOUNTS.get(skill_name, [])
    if skill_name == "component-harness-sopbench" and domain:
        mounts += _sopbench_mounts(domain)
    if extra_mounts:
        mounts += list(extra_mounts)

    seen: set[str] = set()
    for relpath, mode in mounts:
        host_path = (root / relpath).resolve()
        key = str(host_path)
        if key in seen:
            continue
        if not host_path.exists():
            continue
        seen.add(key)
        cmd += ["--mount", _mount_arg(host_path, mode)]

    # Env passthrough.
    for k in os.environ:
        if any(k.startswith(p) for p in _ENV_PASSTHROUGH_PREFIXES) or k in _ENV_PASSTHROUGH_EXACT:
            cmd += ["-e", k]
    if extra_env:
        for k, v in extra_env.items():
            cmd += ["-e", f"{k}={v}"]

    # Entry point = claude binary in the image; args follow.
    cmd += ["--entrypoint", "/usr/bin/claude", DOCKER_IMAGE]
    cmd += claude_args
    return cmd
