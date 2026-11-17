"""Read-only local ROS/source inventory. No publishers, services or robot connections."""
import datetime
import hashlib
import json
import subprocess
import zipfile
from pathlib import Path

def run(args):
    try:
        p = subprocess.run(args, capture_output=True, text=True, timeout=12)
        return {'argv': args, 'returncode': p.returncode, 'stdout': p.stdout, 'stderr': p.stderr}
    except (OSError, subprocess.TimeoutExpired) as e:
        return {'argv': args, 'error': str(e)}

def main():
    out = Path('/mnt/hgfs/vmshare')
    if not out.is_dir():
        raise SystemExit('共享目录不存在，请先连接 /mnt/hgfs/vmshare')
    report = []
    for args in [
        ['ros2','pkg','prefix','rm_driver'],
        ['ros2','pkg','prefix','rm_ros_interfaces'],
        ['ros2','node','list'],
        ['ros2','topic','list','-t'],
    ]:
        report.append(run(args))
    for name in ['Cartepos','Carteposcustom','Stop','Movel']:
        report.append(run(['ros2','interface','show','rm_ros_interfaces/msg/'+name]))
    for topic in ['movep_canfd_cmd','movep_canfd_custom_cmd','move_stop_cmd']:
        report.append(run(['ros2','topic','info','/rm_driver/'+topic,'--verbose']))
    root = Path.home()/'DYS_ws/src/ros2_rm_robot-humble'
    files=[]
    if root.is_dir():
        for package in ['rm_driver','rm_ros_interfaces']:
            folder=root/package
            if not folder.is_dir(): continue
            for f in folder.rglob('*'):
                if f.is_file() and f.suffix.lower() in {'.cpp','.h','.hpp','.msg','.yaml','.xml','.py','.txt','.md'} and f.stat().st_size<2_000_000:
                    files.append(f)
    name='follow_interface_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S')+'.zip'
    with zipfile.ZipFile(out/name,'x',zipfile.ZIP_DEFLATED) as z:
        z.writestr('inventory.json',json.dumps(report,ensure_ascii=False,indent=2))
        z.writestr('source_root.txt',str(root))
        for f in files: z.write(f,'source/'+str(f.relative_to(root)))
        hashes={}
        prefix=report[0].get('stdout','').strip()
        binary=Path(prefix)/'lib/rm_driver/rm_driver'
        if prefix and binary.is_file(): hashes[str(binary)]=hashlib.sha256(binary.read_bytes()).hexdigest()
        z.writestr('binary_hashes.json',json.dumps(hashes,indent=2))
    print('完成：',out/name)
    print('源码文件数：',len(files))
    print('未创建发布器、未调用服务、未发送运动或停止指令。')

if __name__=='__main__': main()
