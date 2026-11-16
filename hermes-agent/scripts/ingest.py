"""批量入库脚本：``python scripts/ingest.py samples``。

CLI 的 ``hermes ingest`` 已覆盖同样能力，这里保留脚本形式方便 CI/批处理调用。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from hermes.api.app import build_state
from hermes.config import Settings


def main() -> int:
    parser = argparse.ArgumentParser(description="把文件或目录入库到 Hermes 知识库")
    parser.add_argument("path", help="文件或目录路径")
    parser.add_argument("--version", default="1")
    parser.add_argument("--doc-id", default=None)
    args = parser.parse_args()

    state = build_state(Settings.from_env())
    target = Path(args.path)
    reports = (state.knowledge.ingest_dir(target, version=args.version) if target.is_dir()
               else [state.knowledge.ingest_path(target, doc_id=args.doc_id, version=args.version)])

    print(json.dumps([report.__dict__ for report in reports], ensure_ascii=False, indent=2))
    print(f"\n共 {len(reports)} 个文档、{sum(r.chunks for r in reports)} 个分块")
    state.db.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
