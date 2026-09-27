import os
import psutil
import time
import pyarrow.parquet as pq

print('--- Processes ---')
target_pid = None
for p in psutil.process_iter(['pid', 'name', 'cmdline', 'cpu_percent', 'memory_info', 'create_time']):
    try:
        cmd = p.info['cmdline']
        if cmd and 'python' in p.info['name'].lower() and any('train.py' in arg for arg in cmd):
            target_pid = p.info['pid']
            print(f"PID: {p.info['pid']}")
            print(f"Cmdline: {' '.join(cmd)}")
            print(f"Start Time: {time.ctime(p.info['create_time'])}")
            print(f"CPU Usage: {p.cpu_percent(interval=1)}%")
            print(f"RAM Usage: {p.info['memory_info'].rss / (1024**3):.2f} GB")
            print('-'*40)
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        pass

print('\n--- Spool Size ---')
spool_path = r'output/.pair_spool/pairs.parquet'
if os.path.exists(spool_path):
    size_mb = os.path.getsize(spool_path) / (1024**2)
    print(f"Spool size: {size_mb:.2f} MB")
    try:
        pf = pq.ParquetFile(spool_path)
        print(f"Spool Row Groups: {pf.metadata.num_row_groups}")
        print(f"Spool Rows Written: {pf.metadata.num_rows:,}")
    except Exception as e:
        print(f"Could not read spool metadata: {e}")
else:
    print('Spool not found')
