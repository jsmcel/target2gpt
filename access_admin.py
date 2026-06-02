#!/usr/bin/env python
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import sys

from target2_access import approve_user, list_latest_access_requests, list_users, revoke_user, set_user_password


def cmd_approve(args: argparse.Namespace) -> int:
    user, key = approve_user(args.email, name=args.name or "", temporary_key=args.key)
    if args.write_key:
        path = Path(args.write_key)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(key + "\n", encoding="utf-8")
        key_line = f"temporary key written to {path}"
    else:
        key_line = f"temporary key: {key}"
    print(f"approved: {user['email']}")
    print(key_line)
    print("user must change the key at first login")
    return 0


def cmd_reset_key(args: argparse.Namespace) -> int:
    user, key = approve_user(args.email, name=args.name or "", temporary_key=args.key)
    if args.write_key:
        path = Path(args.write_key)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(key + "\n", encoding="utf-8")
        key_line = f"temporary key written to {path}"
    else:
        key_line = f"temporary key: {key}"
    print(f"reset key: {user['email']}")
    print(key_line)
    print("user must change the key at next login")
    return 0


def cmd_set_password(args: argparse.Namespace) -> int:
    user = set_user_password(args.email, args.password, must_change_password=args.must_change)
    print(f"password updated: {user['email']}")
    return 0


def cmd_revoke(args: argparse.Namespace) -> int:
    user = revoke_user(args.email)
    print(f"revoked: {user['email']}")
    return 0


def cmd_list(_: argparse.Namespace) -> int:
    users = list_users()
    if not users:
        print("no users")
        return 0
    for user in users:
        status = user.get("status", "unknown")
        must_change = "change-required" if user.get("must_change_password") else "ready"
        print(f"{user.get('email','')} | {status} | {must_change} | {user.get('name','')}")
    return 0


def cmd_requests(_: argparse.Namespace) -> int:
    requests = list_latest_access_requests()
    if not requests:
        print("no access requests")
        return 0
    for item in requests:
        ts = int(item.get("ts") or 0)
        when = datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S") if ts else "unknown"
        status = item.get("approval_status") or "pending"
        must_change = "change-required" if item.get("must_change_password") else ""
        print(
            " | ".join(
                [
                    str(item.get("email") or ""),
                    status,
                    must_change,
                    when,
                    str(item.get("name") or ""),
                    str(item.get("organization") or ""),
                ]
            ).rstrip(" |")
        )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Manage approved target2 GPT users")
    sub = parser.add_subparsers(dest="command", required=True)

    approve = sub.add_parser("approve", help="Approve an email and generate a temporary key")
    approve.add_argument("email")
    approve.add_argument("--name", default="")
    approve.add_argument("--key", default=None, help="Optional explicit temporary key")
    approve.add_argument("--write-key", default=None, help="Write the temporary key to this local file")
    approve.set_defaults(func=cmd_approve)

    reset = sub.add_parser("reset-key", help="Generate a new temporary key for an approved email")
    reset.add_argument("email")
    reset.add_argument("--name", default="")
    reset.add_argument("--key", default=None, help="Optional explicit temporary key")
    reset.add_argument("--write-key", default=None, help="Write the temporary key to this local file")
    reset.set_defaults(func=cmd_reset_key)

    set_password = sub.add_parser("set-password", help="Set a password directly")
    set_password.add_argument("email")
    set_password.add_argument("password")
    set_password.add_argument("--must-change", action="store_true")
    set_password.set_defaults(func=cmd_set_password)

    revoke = sub.add_parser("revoke", help="Revoke an approved email")
    revoke.add_argument("email")
    revoke.set_defaults(func=cmd_revoke)

    list_cmd = sub.add_parser("list", help="List configured users")
    list_cmd.set_defaults(func=cmd_list)

    requests = sub.add_parser("requests", help="List latest access requests")
    requests.set_defaults(func=cmd_requests)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args) or 0)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())


