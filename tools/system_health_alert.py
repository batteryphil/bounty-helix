import psutil
import json
import os

def get_system_health():
    cpu_usage = psutil.cpu_percent()
    vram_usage = psutil.virtual_memory().used / (1024 * 1024 * 1024)
    return {
        "cpu_usage": cpu_usage,
        "vram_usage": vram_usage
    }

def check_system_health():
    health = get_system_health()
    if health["cpu_usage"] > 80:
        return "alert: CPU usage exceeds 80%, current usage: {:.2f}%".format(health["cpu_usage"])
    elif health["vram_usage"] > 10:
        return "alert: VRAM usage exceeds 10GB, current usage: {:.2f}GB".format(health["vram_usage"])
    else:
        return "ok"

if __name__ == "__main__":
    print(check_system_health())