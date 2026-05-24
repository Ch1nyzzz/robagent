"""Pure-Python Toolathlon task orchestrator.

Replaces the Toolathlon-src/scripts/run_single_decoupled.sh shell script so
that the robagent runtime stays self-contained (no upstream patches). The
flow is faithful to the upstream "decoupled" lifecycle:

    1. docker run -d (host network, mount /var/run/docker.sock + dumps + logs)
    2. wait container ready
    3. docker cp project files (configs, scripts, utils, ...) + task dir
    4. (best-effort) install Google credentials into ~/.gmail-mcp etc.
    5. (best-effort) install ./configs/.mcp-auth → /root/.mcp-auth
    6. docker exec container_preprocess
    7. docker exec chown -R host_uid:host_gid (so host-side loop can write)
    8. docker exec container_tool_gateway (background) + wait /health
    9. host: uv run python -m agent_toolathlon.runtime.host_agent_loop
       (with TOOLATHLON_CANDIDATE / TOOLATHLON_OPENAI_* / PYTHONPATH set)
   10. docker exec container_eval
   11. docker exec chown again (eval writes new files)
   12. docker stop && docker rm

Every step that writes into /workspace/dumps re-chowns to the host caller,
so /eval_res.json + status.json end up readable from the host without sudo.
"""
from __future__ import annotations

import os
import shlex
import socket
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path


def find_free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


# Files to copy from Toolathlon-src into the container's /workspace.
# Mirrors FILES_TO_COPY in run_single_decoupled.sh (line ~264).
_FILES_TO_COPY = [
    "configs",
    "deployment/k8s",
    "scripts",
    "deployment/canvas/logs",
    "global_preparation/check_installation.py",
    "local_binary/github-mcp-server",
    "utils",
    "main.py",
]


@dataclass
class OrchestratorConfig:
    toolathlon_src: Path              # vendored repo root
    robagent_root: Path               # for PYTHONPATH (imports agent_toolathlon, bench)
    candidate: str                    # agent_toolathlon.<candidate>.agent
    image: str
    output_folder: Path               # per-task host dump dir, must exist
    logs_folder: Path                 # per-task host log dir, must exist
    runmode: str                      # "quickstart" or "normal"
    model: str
    provider: str
    max_steps: int
    eval_config: str = "scripts/formal_run_v0.json"
    openai_base_url: str = ""
    openai_api_key: str = ""
    host_loop_module: str = "agent_toolathlon.runtime.host_agent_loop"
    container_name: str = ""          # filled in run_task if empty
    debug: bool = True
    gateway_health_timeout_sec: int = 40


def _docker(args: list[str], **kw) -> subprocess.CompletedProcess:
    return subprocess.run(["docker", *args], **kw)


def _exec(container: str, cmd: str | list[str], *, env: dict[str, str] | None = None,
          stdout=None, stderr=None, check: bool = False) -> int:
    args = ["exec"]
    args += ["--env", "DOCKER_API_VERSION=1.44"]
    if env:
        for k, v in env.items():
            args += ["--env", f"{k}={v}"]
    args.append(container)
    if isinstance(cmd, str):
        args += ["bash", "-lc", cmd]
    else:
        args += cmd
    p = _docker(args, stdout=stdout, stderr=stderr)
    if check and p.returncode != 0:
        raise RuntimeError(f"docker exec failed (rc={p.returncode}): {cmd}")
    return p.returncode


def _exec_detached(container: str, sh: str) -> None:
    _docker(
        ["exec", "--env", "DOCKER_API_VERSION=1.44", "-d", container, "bash", "-lc", sh],
        check=False,
    )


def _wait_container_ready(container: str, timeout: int = 30) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if _docker(
            ["ps", "-q", "--filter", f"name={container}"],
            capture_output=True, text=True,
        ).stdout.strip():
            if _exec(container, "echo ready", stdout=subprocess.DEVNULL,
                     stderr=subprocess.DEVNULL) == 0:
                return True
        time.sleep(1)
    return False


def _wait_gateway(port: int, timeout: int) -> bool:
    import urllib.request
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(
                f"http://127.0.0.1:{port}/health", timeout=2
            ) as r:
                if r.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(1)
    return False


def _copy_into_workspace(container: str, src: Path, dst_under_workspace: str) -> None:
    if not src.exists():
        return
    if src.is_dir():
        parent = os.path.dirname(dst_under_workspace.rstrip("/"))
        if parent:
            _exec(container, f"mkdir -p /workspace/{parent}")
    _docker(["cp", str(src), f"{container}:/workspace/{dst_under_workspace}"],
            check=False)


def _try_install_gcp_creds(container: str, src_configs: Path, runmode: str) -> None:
    """Mirrors the copy_config_cmd block in the shell script."""
    oauth = src_configs / "gcp-oauth.keys.json"
    creds = src_configs / "google_credentials.json"
    if not (oauth.exists() and creds.exists()):
        if runmode == "quickstart":
            return
        # Non-quickstart with missing creds → caller should know.
        return
    cmd = (
        "for dir in ~/.gmail-mcp ~/.calendar-mcp; do "
        "mkdir -p $dir && "
        "cp ./configs/gcp-oauth.keys.json $dir/ && "
        "cp ./configs/google_credentials.json $dir/credentials.json; done"
    )
    _exec(container, cmd, check=False)


def _try_install_mcp_auth(container: str, src_configs: Path) -> None:
    mcp_auth = src_configs / ".mcp-auth"
    if not mcp_auth.exists():
        return
    _exec(container, "mkdir -p /root/.mcp-auth", check=False)
    _docker(["cp", f"{str(mcp_auth)}/.", f"{container}:/root/.mcp-auth/"],
            check=False)


def _docker_run_container(cfg: OrchestratorConfig) -> None:
    """Start the task container; uses the same args as run_single_decoupled.sh."""
    args = [
        "run", "-d", "--name", cfg.container_name,
        "--network", "host",
        "-v", "/var/run/docker.sock:/var/run/docker.sock",
        "-v", f"{cfg.output_folder.resolve()}:/workspace/dumps",
        "-v", f"{cfg.logs_folder.resolve()}:/workspace/logs",
        "-w", "/workspace",
    ]
    if cfg.openai_base_url:
        args += ["-e", f"TOOLATHLON_OPENAI_BASE_URL={cfg.openai_base_url}"]
    if cfg.openai_api_key:
        args += ["-e", f"TOOLATHLON_OPENAI_API_KEY={cfg.openai_api_key}"]
    args += [cfg.image, "sleep", "3600"]
    p = _docker(args, capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(
            f"docker run failed: {p.stderr.strip()}"
        )


def _chown_dumps(container: str) -> None:
    uid, gid = os.getuid(), os.getgid()
    _exec(
        container,
        f"chown -R {uid}:{gid} /workspace/dumps /workspace/logs 2>/dev/null || true",
    )


def run_task(
    *,
    task_dir_arg: str,            # e.g. "finalpool/find-alita-paper"
    cfg: OrchestratorConfig,
    host_log_path: Path | None = None,
) -> dict:
    """Run a single Toolathlon task end-to-end. Returns a result dict.

    The container is destroyed at the end regardless of success/failure.
    """
    if not cfg.container_name:
        ts = time.strftime("%Y%m%d-%H%M%S")
        safe = task_dir_arg.replace("/", "-")
        cfg.container_name = f"robagent-toolathlon-{safe}-{ts}-{os.getpid()}"

    cfg.output_folder.mkdir(parents=True, exist_ok=True)
    cfg.logs_folder.mkdir(parents=True, exist_ok=True)

    started = time.time()
    host_loop_rc = None
    eval_rc = None
    error: str | None = None

    def _log(msg: str) -> None:
        line = f"[orchestrator] {msg}"
        if host_log_path is not None:
            with host_log_path.open("a", encoding="utf-8") as fh:
                fh.write(line + "\n")
        # Also print summarized progress.
        print(line, flush=True)

    try:
        _log(f"step 1: docker run {cfg.container_name}")
        _docker_run_container(cfg)
        if not _wait_container_ready(cfg.container_name):
            raise RuntimeError("container did not become ready")

        _log("step 2.5: copy project files into /workspace")
        # Ensure parent dirs exist
        for parent in ("deployment", "deployment/canvas", "global_preparation", "tasks"):
            _exec(cfg.container_name, f"mkdir -p /workspace/{parent}")
        for item in _FILES_TO_COPY:
            _copy_into_workspace(cfg.container_name, cfg.toolathlon_src / item, item)

        # Copy the task directory
        taskdomain, taskname = task_dir_arg.split("/", 1)
        _exec(cfg.container_name, f"mkdir -p /workspace/tasks/{taskdomain}")
        _copy_into_workspace(
            cfg.container_name,
            cfg.toolathlon_src / "tasks" / task_dir_arg,
            f"tasks/{taskdomain}/",
        )

        _log("step 2.6: install gcp creds + mcp-auth if present")
        _try_install_gcp_creds(cfg.container_name, cfg.toolathlon_src / "configs", cfg.runmode)
        _try_install_mcp_auth(cfg.container_name, cfg.toolathlon_src / "configs")

        _log("step 3: preprocess")
        preprocess_cmd = (
            f"cd /workspace && uv run python -m scripts.decoupled.container_preprocess "
            f"--eval_config {cfg.eval_config} "
            f"--task_dir {task_dir_arg} "
            f"--max_steps_under_single_turn_mode {cfg.max_steps} "
            f"--model_short_name {cfg.model} "
            f"--provider {cfg.provider} "
            f"--bundle_file /workspace/dumps/task_bundle.json "
            f"--host_output_folder {cfg.output_folder.resolve()} "
            f"--debug "
            f"> /workspace/logs/preprocess.log 2>&1"
        )
        rc = _exec(cfg.container_name, preprocess_cmd)
        if rc != 0:
            # Pull preprocess.log out for the host caller.
            raise RuntimeError(
                f"preprocess failed (rc={rc}); see {cfg.logs_folder}/preprocess.log"
            )

        bundle_file = cfg.output_folder / "task_bundle.json"
        if not bundle_file.exists():
            raise RuntimeError(f"missing bundle file after preprocess: {bundle_file}")

        _log("step 3.5: chown dumps to host uid")
        _chown_dumps(cfg.container_name)

        _log("step 4: start MCP gateway")
        port = find_free_port()
        gateway_sh = (
            f"cd /workspace && nohup uv run python -m scripts.decoupled.container_tool_gateway "
            f"--bundle_file /workspace/dumps/task_bundle.json "
            f"--host 0.0.0.0 --port {port} --debug "
            f"> /workspace/logs/gateway.log 2>&1 &"
        )
        _exec_detached(cfg.container_name, gateway_sh)
        if not _wait_gateway(port, cfg.gateway_health_timeout_sec):
            raise RuntimeError(f"gateway did not become healthy on :{port}")

        _log(f"step 5: host agent loop (candidate={cfg.candidate})")
        env = os.environ.copy()
        env["PYTHONPATH"] = str(cfg.robagent_root) + os.pathsep + env.get("PYTHONPATH", "")
        env["TOOLATHLON_CANDIDATE"] = cfg.candidate
        if cfg.openai_base_url:
            env["TOOLATHLON_OPENAI_BASE_URL"] = cfg.openai_base_url
        if cfg.openai_api_key:
            env["TOOLATHLON_OPENAI_API_KEY"] = cfg.openai_api_key
        # Forward component-runtime env vars to the host-loop subprocess.
        # `uv run --directory ...` may scrub some inherited vars, so we
        # set them explicitly. Absolute paths in COMPONENT_FILES survive
        # the cwd change; relative ones would not.
        for k in (
            "COMPONENT_WORKFLOW",
            "COMPONENT_DIR",
            "COMPONENT_NAMES",
            "COMPONENT_FILES",
            "COMPONENT_RUN_TAG",
            "COMPONENT_STATE_DIR",
        ):
            if k in os.environ:
                env[k] = os.environ[k]

        host_loop_cmd = [
            "uv", "run", "--directory", str(cfg.toolathlon_src),
            "python", "-m", cfg.host_loop_module,
            "--bundle_file", str(bundle_file.resolve()),
            "--gateway_url", f"http://127.0.0.1:{port}/sse",
            "--gateway_server_name", "gw",
            "--debug",
        ]
        host_loop_log = cfg.logs_folder / "host_loop.log"
        with host_loop_log.open("w", encoding="utf-8") as fh:
            host_loop_rc = subprocess.run(
                host_loop_cmd, env=env, cwd=str(cfg.toolathlon_src),
                stdout=fh, stderr=subprocess.STDOUT,
            ).returncode
        _log(f"step 5 done: host_loop_rc={host_loop_rc}")

        _log("step 6: eval")
        eval_cmd = (
            f"cd /workspace && uv run python -m scripts.decoupled.container_eval "
            f"--bundle_file /workspace/dumps/task_bundle.json "
            f"> /workspace/logs/eval.log 2>&1"
        )
        eval_rc = _exec(cfg.container_name, eval_cmd)
        _log(f"step 6 done: eval_rc={eval_rc}")

        _log("step 6.5: chown dumps to host uid again")
        _chown_dumps(cfg.container_name)

    except Exception as e:
        error = repr(e)
        _log(f"ERROR: {error}")
    finally:
        # Best-effort: collect container.log before cleanup
        try:
            with (cfg.logs_folder / "container.log").open("w", encoding="utf-8") as fh:
                _docker(["logs", cfg.container_name], stdout=fh, stderr=subprocess.STDOUT,
                        check=False)
        except Exception:
            pass

        # Cleanup container
        _docker(["stop", cfg.container_name],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
        _docker(["rm", cfg.container_name],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)

    return {
        "container_name": cfg.container_name,
        "elapsed_sec": round(time.time() - started, 1),
        "host_loop_rc": host_loop_rc,
        "eval_rc": eval_rc,
        "error": error,
    }
