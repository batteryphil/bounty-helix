import psutil
import json

def check_system_health():
    cpu_usage = psutil.cpu_percent()
    vram_usage = psutil.virtual_memory().total - psutil.virtual_memory().available

    if cpu_usage > 80:
        return json.dumps({"status": "alert", "message": f"CPU usage exceeded 80%: {cpu_usage}%"})
    elif vram_usage > 10 * 1024 * 1024 * 1024:  # 10 GB
        return json.dumps({"status": "alert", "message": f"VRAM usage exceeded 10GB: {vram_usage / (1024 * 1024 * 1024):.2f} GB"})
    else:
        return json.dumps({"status": "ok", "message": "system is ok"})

if __name__ == '__main__':
    print(check_system_health())