"""Build minepy 1.2.6 with a regenerated Cython wrapper for Python 3.11.

Run from the activated venv using install_minepy.ps1. Requires Microsoft C++
Build Tools with the x64 compiler and a Windows SDK. No project code is run.
"""
from pathlib import Path
import hashlib,json,subprocess,sys,tarfile,tempfile,urllib.request
ROOT=Path(__file__).resolve().parents[1]


def run(*args,cwd=None):
    print("INSTALL",*args,flush=True);subprocess.run([sys.executable,*args],cwd=cwd,check=True)


def main():
    if sys.prefix==sys.base_prefix:raise RuntimeError("Activate the project's .venv before installing minepy.")
    if sys.version_info[:2]!=(3,11):raise RuntimeError("This recipe targets Python 3.11.")
    run("-m","pip","install","pip==24.3.1","setuptools==75.3.0","wheel==0.44.0","Cython==0.29.37","numpy==1.26.4")
    with urllib.request.urlopen("https://pypi.org/pypi/minepy/1.2.6/json",timeout=60) as response:meta=json.load(response)
    sources=[r for r in meta["urls"] if r["packagetype"]=="sdist"]
    if len(sources)!=1:raise RuntimeError("Expected exactly one official minepy 1.2.6 source distribution.")
    source=sources[0];wheels=ROOT/"wheels";wheels.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="minepy build ") as tmp:
        directory=Path(tmp);archive=directory/"minepy.tar.gz"
        with urllib.request.urlopen(source["url"],timeout=120) as response:payload=response.read()
        digest=hashlib.sha256(payload).hexdigest()
        if digest!=source["digests"]["sha256"]:raise RuntimeError("minepy download checksum mismatch.")
        archive.write_bytes(payload)
        with tarfile.open(archive,"r:gz") as stream:
            for member in stream.getmembers():
                target=(directory/member.name).resolve()
                if not target.is_relative_to(directory.resolve()) or member.issym() or member.islnk():raise RuntimeError("Unsafe archive member.")
            stream.extractall(directory,filter="data")
        folder=directory/"minepy-1.2.6"
        run("-m","cython","-3","minepy/mine.pyx",cwd=folder)
        run("-m","pip","wheel","--no-build-isolation","--no-deps","--wheel-dir",str(wheels),str(folder))
        candidates=list(wheels.glob("minepy-1.2.6-*.whl"))
        compatible=[p for p in candidates if "cp311" in p.name]
        if not compatible:raise RuntimeError("No CPython 3.11 minepy wheel was produced.")
        wheel=max(compatible,key=lambda p:p.stat().st_mtime_ns)
        run("-m","pip","install","--force-reinstall","--no-deps",str(wheel))
        receipt={"source_sha256":digest,"minepy":"1.2.6","Cython":"0.29.37","setuptools":"75.3.0","wheel":wheel.name,"wheel_sha256":hashlib.sha256(wheel.read_bytes()).hexdigest()}
        (wheels/"minepy_build_receipt.json").write_text(json.dumps(receipt,indent=2)+"\n")
    run("-c","from minepy import MINE; import numpy as np; m=MINE(); x=np.arange(32,dtype=float); m.compute_score(x,x); print('minepy MIC probe:',m.mic())")
    return 0


if __name__=="__main__":raise SystemExit(main())
