"""进程级恢复演练：真的 kill 服务进程，验证启动对账与 checkpoint 续跑。

用法：
    python scripts/recovery_drill.py
    python scripts/recovery_drill.py --delay 1.5 --port 8023 --json

流程：
1. 用 Mock Provider 起服务，每次 LLM 调用人为延迟，制造可被 kill 的窗口；
2. 提交一个「计算 21 + 21」的两步 run；
3. 等到第 1 步 checkpoint 落盘后 kill -9；
4. 重启服务，断言 run 被标记为 interrupted（process_restart + recoverable）；
5. 调 /resume，断言最终 succeeded，且工具只执行一次。
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx

TERMINAL_STATES = {"succeeded", "failed", "cancelled"}
TIMELINE_TYPES = {
    "run.created", "run.started", "run.failed", "run.resumed", "run.completed",
    "llm.request", "step.completed", "tool.completed",
}


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _child_env(db_path: str, delay: float) -> dict[str, str]:
    env = os.environ.copy()
    env.update({
        "HERMES_LLM_PROVIDER": "mock",
        "HERMES_MOCK_DELAY_S": str(delay),
        "HERMES_DB_PATH": db_path,
        "HERMES_TRIGGER_TICK_S": "0",
        "HERMES_RERANK": "rrf",  # 避免本机装了 semantic extra 时把重排模型一起加载
        "HERMES_AUTH_MODE": "off",
    })
    return env


def _start_server(port: int, env: dict[str, str]) -> subprocess.Popen:
    return subprocess.Popen(
        [sys.executable, "-m", "hermes.cli", "serve", "--port", str(port)],
        env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )


def _wait_health(client: httpx.Client, timeout_s: float = 30.0) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            if client.get("/healthz").status_code == 200:
                return
        except httpx.HTTPError:
            pass
        time.sleep(0.2)
    raise RuntimeError(f"服务未在 {timeout_s:.0f}s 内就绪")


def _wait_checkpoint(client: httpx.Client, run_id: str, timeout_s: float) -> int:
    deadline = time.time() + timeout_s
    steps = 0
    while time.time() < deadline:
        steps = client.get(f"/v1/runs/{run_id}").json().get("steps") or 0
        if steps >= 1:
            return steps
        time.sleep(0.05)
    raise RuntimeError(f"run 在 {timeout_s:.0f}s 内没有产生 checkpoint（steps={steps}）")


def _wait_state(client: httpx.Client, run_id: str, expected: set[str],
                timeout_s: float) -> str:
    deadline = time.time() + timeout_s
    state = ""
    while time.time() < deadline:
        state = client.get(f"/v1/runs/{run_id}").json()["state"]
        if state in expected:
            return state
        time.sleep(0.1)
    raise RuntimeError(f"等待状态 {sorted(expected)} 超时，当前 {state}")


def drill(port: int, db_path: str, delay: float) -> dict:
    for suffix in ("", "-wal", "-shm", "-journal"):
        Path(db_path + suffix).unlink(missing_ok=True)
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)

    env = _child_env(db_path, delay)
    proc = _start_server(port, env)
    kill_ts = 0.0
    try:
        with httpx.Client(base_url=f"http://127.0.0.1:{port}", timeout=30.0) as client:
            _wait_health(client)
            run_id = client.post("/v1/runs", json={"input": "计算 21 + 21"}).json()["run_id"]
            steps_before_kill = _wait_checkpoint(client, run_id, timeout_s=delay * 4 + 15)

            kill_ts = time.time()
            proc.kill()
            proc.wait(timeout=15)

            proc = _start_server(port, env)
            _wait_health(client)
            _wait_state(client, run_id, {"interrupted"}, timeout_s=30)

            response = client.post(f"/v1/runs/{run_id}/resume")
            if response.status_code != 200:
                raise RuntimeError(f"resume 失败：{response.status_code} {response.text}")
            final_state = _wait_state(client, run_id, TERMINAL_STATES, timeout_s=60)

            row = client.get(f"/v1/runs/{run_id}").json()
            events = client.get(f"/v1/runs/{run_id}/trace").json()["events"]
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=15)

    restart_events = [e for e in events
                      if e["payload"].get("kind") == "process_restart"]
    tool_runs = [e for e in events if e["type"] == "tool.completed"]
    checks = {
        "final_succeeded": final_state == "succeeded",
        "marked_interrupted": bool(restart_events),
        "recoverable": bool(restart_events) and bool(restart_events[0]["payload"].get("recoverable")),
        "resumed_event": any(e["type"] == "run.resumed" for e in events),
        "tool_executed_once": len(tool_runs) == 1,
        "answer_has_result": "42" in (row.get("answer") or ""),
    }
    return {
        "run_id": run_id,
        "port": port,
        "delay_s": delay,
        "steps_before_kill": steps_before_kill,
        "final_state": final_state,
        "tool_calls": len(tool_runs),
        "checks": checks,
        "passed": all(checks.values()),
        "timeline": _timeline(events, kill_ts),
    }


def _timeline(events: list[dict], kill_ts: float) -> list[str]:
    if not events:
        return []
    origin = events[0]["ts"]
    lines = []
    killed = False
    for event in events:
        if event["type"] not in TIMELINE_TYPES:
            continue
        if not killed and kill_ts and event["ts"] > kill_ts:
            lines.append("        -- kill -9，进程退出 --")
            killed = True
        offset = (event["ts"] - origin) * 1000
        lines.append(f"+{offset:7.0f}ms  {event['type']}")
    return lines


def main() -> int:
    parser = argparse.ArgumentParser(description="进程级恢复演练（kill -9 → 重启对账 → resume）")
    parser.add_argument("--port", type=int, default=0, help="服务端口，0 表示自动挑空闲端口")
    parser.add_argument("--db", default="data/recovery_drill.db", help="演练用 SQLite 路径")
    parser.add_argument("--delay", type=float, default=2.0, help="每次 Mock LLM 调用的人为延迟（秒）")
    parser.add_argument("--json", action="store_true", help="以 JSON 输出结果")
    args = parser.parse_args()

    port = args.port or _free_port()
    try:
        result = drill(port, args.db, args.delay)
    except Exception as exc:  # noqa: BLE001 演练脚本要把失败原因完整打出来
        print(f"恢复演练失败：{exc}")
        return 1

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print("进程级恢复演练（Mock，LLM 延迟 {:.1f}s）".format(result["delay_s"]))
        print(f"- run：{result['run_id']}（端口 {result['port']}）")
        print(f"- kill 时已完成步数：{result['steps_before_kill']}")
        print(f"- 重启对账：interrupted → resume → {result['final_state']}")
        print(f"- 工具执行次数：{result['tool_calls']}（续跑没有重复执行）")
        print("- 时间线：")
        for line in result["timeline"]:
            print(f"    {line}")
        print(f"- 结论：{'通过' if result['passed'] else '失败'}")
        failed = [name for name, ok in result["checks"].items() if not ok]
        if failed:
            print(f"  未通过的检查：{', '.join(failed)}")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
