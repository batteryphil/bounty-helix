import psutil
import json

def get_system_health():
    cpu_usage = psutil.cpu_percent()
    vram_usage = psutil.virtual_memory().used / (1024 * 1024 * 1024)
    return {
        'cpu_usage': cpu_usage if cpu_usage > 80 else 'ok',
        'vram_usage': 'ok' if vram_usage < 10 else f'{vram_usage:.2f} GB'
    }

def main():
    health = get_system_health()
    print(json.dumps(health))

if __name__ == '__main__':
    main()