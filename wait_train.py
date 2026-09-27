import os
import psutil
import time

def get_train_pid():
    for p in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            cmd = p.info['cmdline']
            if cmd and 'python' in p.info['name'].lower() and any('train.py' in arg for arg in cmd):
                return p.info['pid']
        except Exception:
            pass
    return None

target_pid = get_train_pid()

if not target_pid:
    print("train.py is NOT running.")
else:
    print(f"Monitoring train.py (PID {target_pid})...")
    while psutil.pid_exists(target_pid):
        spool = 'output/.pair_spool/pairs.parquet'
        size_mb = os.path.getsize(spool)/1024**2 if os.path.exists(spool) else 0
        try:
            p = psutil.Process(target_pid)
            cpu = p.cpu_percent(interval=1)
            ram = p.memory_info().rss / 1024**3
            print(f"[{time.strftime('%H:%M:%S')}] CPU: {cpu:5.1f}% | RAM: {ram:5.2f} GB | Spool: {size_mb:6.1f} MB")
        except Exception as e:
            pass
        time.sleep(300)  # Check every 5 minutes
    
    print("train.py has TERMINATED.")
