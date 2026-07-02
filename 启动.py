"""
车型图像识别系统 — 一键启动脚本
启动 FastAPI 后端服务（含前端静态文件）
"""
import subprocess
import webbrowser
import time
import os
import sys

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.join(PROJECT_DIR, "backend")

def main():
    print("=" * 50)
    print("  车型图像识别系统 — 启动中...")
    print("=" * 50)
    print(f"  项目目录: {PROJECT_DIR}")
    print()

    # 启动后端
    print("[1/2] 启动后端服务 (FastAPI + 三模型)...")
    server_cmd = [
        sys.executable, "-m", "uvicorn",
        "main:app",
        "--host", "0.0.0.0",
        "--port", "8080",
    ]

    server_proc = subprocess.Popen(
        server_cmd,
        cwd=BACKEND_DIR,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
    )

    # 等待服务启动
    print("  等待模型加载...")
    time.sleep(3)

    # 打开浏览器
    print("[2/2] 打开浏览器...")
    url = "http://localhost:8080"
    webbrowser.open(url)

    print()
    print("=" * 50)
    print(f"  ✅ 服务已启动: {url}")
    print("  📸 上传车辆图片即可开始识别")
    print("  🛑 按 Ctrl+C 停止服务")
    print("=" * 50)
    print()

    try:
        # 持续打印服务日志
        for line in server_proc.stdout:
            print(line, end="")
    except KeyboardInterrupt:
        print("\n正在关闭服务...")
        server_proc.terminate()
        server_proc.wait()
        print("服务已停止。")


if __name__ == "__main__":
    main()
