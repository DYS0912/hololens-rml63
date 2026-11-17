"""Collect local SDK/build inputs; never start drivers or contact the robot."""
from pathlib import Path
import datetime
import json
import subprocess
import zipfile

def main():
    root=Path.home()/'DYS_ws/src/ros2_rm_robot-humble'
    out=Path('/mnt/hgfs/vmshare')
    if not root.is_dir() or not out.is_dir():
        raise SystemExit('源码目录或共享目录不存在，未修改任何文件')
    report=[]
    for cmd in [['uname','-a'],['g++','--version'],['cmake','--version'],['ldd','--version']]:
        try:
            p=subprocess.run(cmd,capture_output=True,text=True,timeout=10)
            report.append(dict(command=cmd,code=p.returncode,stdout=p.stdout,stderr=p.stderr))
        except (OSError,subprocess.TimeoutExpired) as e:
            report.append(dict(command=cmd,error=str(e)))
    path=out/('follow_build_inputs_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S')+'.zip')
    count=0
    with zipfile.ZipFile(path,'x',zipfile.ZIP_DEFLATED) as z:
        z.writestr('environment.json',json.dumps(report,indent=2))
        for name in ['rm_driver','rm_ros_interfaces']:
            folder=root/name
            for f in folder.rglob('*'):
                if not f.is_file() or '.git' in f.parts: continue
                # Include the SDK binary omitted by the earlier source-only collector.
                if f.stat().st_size>150_000_000: continue
                z.write(f,'source/'+str(f.relative_to(root))); count+=1
    print('完成：',path)
    print('文件数：',count)
    print('仅打包本地构建资料，未启动驱动，未发送机器人指令。')

if __name__=='__main__': main()
