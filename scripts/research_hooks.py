#!/usr/bin/env python3
"""Portable stdlib-only hook/agent entry point. No package install or network needed."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from academic_helper.research_delivery import DeliveryError, handle_hook, record, refresh


def main() -> int:
    # Hook transports expect UTF-8 JSON, including Windows pipes and Chinese paths.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="助手用：研究入口與交付紀錄")
    commands = parser.add_subparsers(dest="action", required=True)
    commands.add_parser("hook")
    refresh_parser = commands.add_parser("refresh")
    refresh_parser.add_argument("--workspace", required=True)
    register = commands.add_parser("record")
    register.add_argument("--workspace", required=True)
    register.add_argument("--task", required=True)
    register.add_argument("--purpose", required=True)
    register.add_argument("--status", choices=["ready", "partial", "failed"], required=True)
    register.add_argument("--primary")
    register.add_argument("--artifact", action="append", default=[])
    register.add_argument("--limitation", action="append", default=[])
    args = parser.parse_args()
    try:
        if args.action == "hook":
            raw = sys.stdin.buffer.read(1024 * 1024 + 1)
            if len(raw) > 1024 * 1024:
                raise DeliveryError("Hook 輸入過大，本次未更新入口。")
            payload = json.loads(raw)
            if not isinstance(payload, dict):
                raise DeliveryError("Hook 輸入格式不支援。")
            result = handle_hook(payload)
        elif args.action == "refresh":
            result = refresh(args.workspace)
        else:
            result = record(args.workspace, args.task, args.purpose, args.status,
                            args.primary, args.artifact, args.limitation)
        print(json.dumps(result, ensure_ascii=False))
        return 0
    except (DeliveryError, OSError, ValueError, TypeError):
        # Never echo tool payloads, credential values, raw exceptions or private content.
        if args.action == "hook":
            print(json.dumps({"systemMessage": "AcademicHelper 入口整理尚未完成；助手可檢查交付紀錄、路徑與入口標記。原始成果保留。"}, ensure_ascii=False))
            return 0
        print(json.dumps({"status": "needs_attention", "message": "請檢查交付紀錄、現有檔案、路徑與入口標記；原始成果保留。"}, ensure_ascii=False))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
