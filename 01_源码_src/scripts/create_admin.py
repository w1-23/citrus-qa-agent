#!/usr/bin/env python3
"""Interactively create the first CitrusQA administrator on a private server session."""
from __future__ import annotations

import getpass
import sys
from pathlib import Path

SOURCE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE_ROOT))

from src.auth import create_first_admin, init_store, list_users  # noqa: E402


def main() -> int:
    init_store()
    if any(user.get("role") == "admin" for user in list_users()):
        print("管理员账号已经存在；本脚本不会创建第二个管理员。")
        return 2

    username = input("管理员用户名（3–32 位）: ").strip()
    password = getpass.getpass("管理员密码（至少 10 位，不会回显）: ")
    confirmation = getpass.getpass("再次输入管理员密码: ")
    if password != confirmation:
        print("两次密码不一致，没有创建账号。")
        return 2

    user, _session_token, error = create_first_admin(username, password)
    if error or not user:
        print(error or "管理员账号创建失败。")
        return 2
    if user.get("role") != "admin":
        print("注册未获得管理员身份；请联系服务器管理员检查账号库。")
        return 3
    print(f"管理员账号 {user['username']} 已创建。请在 HTTPS 网站上登录。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
