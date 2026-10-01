"""Local background collection of the dependency-gated Slurm report artifact."""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import shlex
import subprocess
import tarfile
import time

ROOT=Path(__file__).resolve().parents[3]
REMOTE_REPO="/home/xmz/expert"
RESULT_ROOT="results/pretrain/20261001"


def log_connection(message):
    stamp=datetime.now().astimezone().isoformat(timespec="seconds")
    with (ROOT/"docs/agents/gpu.md").open("a",encoding="utf-8") as stream:
        stream.write(f"\n- {stamp}: Automated pretrain collector: {message}\n")


def save_status(status, **details):
    path=ROOT/"good_result/pretrain/status.json"
    path.parent.mkdir(parents=True,exist_ok=True)
    payload={"status":status,"updated":datetime.now().astimezone().isoformat(),**details}
    temporary=path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    temporary.replace(path)


def extract_delivery(archive,root):
    root=Path(root).resolve()
    allowed_suffixes={".json",".jsonl",".html",".png",".svg"}
    with tarfile.open(archive) as tar:
        members=tar.getmembers()
        for member in members:
            path=PurePosixPath(member.name)
            if (not member.isfile() or path.is_absolute() or ".." in path.parts
                or path.suffix not in allowed_suffixes
                or not (member.name.startswith("good_result/pretrain/")
                        or member.name.startswith(RESULT_ROOT+"/runs/"))
                or not (root/member.name).resolve().is_relative_to(root)):
                raise ValueError(f"unexpected delivery archive member: {member.name}")
        tar.extractall(root,members=members,filter="data")


def collect(args):
    remote_archive=f"{REMOTE_REPO}/{RESULT_ROOT}/delivery.tar.gz"
    if not re.fullmatch(r"/home/xmz/expert-data/[A-Za-z0-9_.-]+\.bundle",args.bundle):
        raise ValueError("collector requires an explicit task-owned Git bundle")
    command=(f"set -e; test -d {REMOTE_REPO}; cd {REMOTE_REPO}; "
             f"git pull --ff-only {shlex.quote(args.bundle)} main >/dev/null; "
             f"if test -f {remote_archive}; then printf 'READY '; sha256sum {remote_archive}; "
             f"else squeue -h -j {args.report_job} -o '%T|%R'; fi")
    deadline=time.monotonic()+args.max_hours*3600
    failures=0
    while time.monotonic()<deadline:
        log_connection(f"pull-first from {args.bundle}; read report job {args.report_job} status "
                       "or final archive checksum only; no computation/source edits/proxy changes.")
        result=subprocess.run([args.ssh,"-o","ClearAllForwardings=yes","-o","ConnectTimeout=20",
                               "Guqq",command],capture_output=True,text=True,timeout=60)
        if result.returncode:
            failures+=1
            if failures>=3 or args.once:
                (ROOT/"docs/agents/lessons.md").read_text(encoding="utf-8")
                raise RuntimeError(f"collector SSH failed after {failures} attempts: {result.stderr[-1500:]}")
            save_status("retrying_connection",attempt=failures,error=result.stderr[-1500:])
            time.sleep(min(args.interval,60))
            continue
        failures=0
        ready=re.search(r"READY ([a-f0-9]{64})",result.stdout)
        if ready:
            local=ROOT/RESULT_ROOT/"delivery.tar.gz"
            local.parent.mkdir(parents=True,exist_ok=True)
            log_connection(f"SCP final report archive {remote_archive} after successful pull-first check; "
                           "no model weights, environments, or unrelated results transferred.")
            subprocess.run([args.scp,"-o","ClearAllForwardings=yes","-o","ConnectTimeout=20",
                            "-o","ServerAliveInterval=30","Guqq:"+remote_archive,str(local)],
                           check=True,timeout=600)
            if hashlib.sha256(local.read_bytes()).hexdigest()!=ready[1]:
                raise ValueError("delivery archive checksum mismatch")
            extract_delivery(local,ROOT)
            audit=json.loads((ROOT/"good_result/pretrain/audit.json").read_text())
            if audit["status"]!="48/48 completed runs":
                raise ValueError("delivery is not the complete audited experiment grid")
            save_status("downloaded_pending_visual_review",archive_sha256=ready[1],report_job=args.report_job)
            stamp=datetime.now().astimezone().isoformat(timespec="minutes")
            with (ROOT/"docs/agents/state.md").open("a",encoding="utf-8") as stream:
                stream.write(f"\n- {stamp}: Pretrain collector downloaded and hash-verified the full "
                             "48-run audited report into good_result/pretrain. Final visual review remains.\n")
            return
        state=result.stdout.strip()
        if not state or "DependencyNeverSatisfied" in state:
            raise RuntimeError("report job absent or dependency failed; agent review required: "+state)
        save_status("running",report_job=args.report_job,scheduler=state)
        if args.once:
            return
        time.sleep(args.interval)
    raise TimeoutError("collector deadline reached; experiment results remain on server")


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--report-job",type=int,required=True)
    parser.add_argument("--bundle",required=True)
    parser.add_argument("--ssh",default="ssh")
    parser.add_argument("--scp",default="scp")
    parser.add_argument("--interval",type=int,default=600)
    parser.add_argument("--max-hours",type=int,default=336)
    parser.add_argument("--once",action="store_true")
    args=parser.parse_args()
    try:
        collect(args)
    except Exception as error:
        save_status("needs_attention",error=str(error),report_job=args.report_job)
        raise


if __name__=="__main__":
    main()
