"""The six paper experiments in order; no training unless --run is explicit."""
import argparse
from datetime import datetime
from pathlib import Path
import subprocess,sys
CODE=Path(__file__).resolve().parents[1]
if __name__=='__main__':
    p=argparse.ArgumentParser();g=p.add_mutually_exclusive_group();g.add_argument('--dry-run',action='store_true');g.add_argument('--run',action='store_true')
    p.add_argument('--batch');p.add_argument('--workers',type=int,choices=range(1,5),default=1);p.add_argument('--max-hours',type=float,default=24);p.add_argument('--resume',action='store_true');p.add_argument('--verify-archived',action='store_true');a=p.parse_args()
    batch=a.batch or datetime.now().strftime('replay_%Y%m%d_%H%M%S')
    for i,n in enumerate(('parameters','mechanisms','statistics','across_r','spatial','initialization'),1):
        cmd=[sys.executable,'-B',str(CODE/'scripts'/f'run_{i:02d}_{n}.py'),'--run' if a.run else '--dry-run','--batch',batch,'--workers',str(a.workers),'--max-hours',str(a.max_hours)]
        if a.resume:cmd+=['--resume']
        if a.verify_archived:cmd+=['--verify-archived']
        subprocess.run(cmd,check=True,cwd=CODE)
