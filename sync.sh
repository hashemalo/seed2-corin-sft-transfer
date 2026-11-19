#!/bin/bash
# Pod helpers. Needs seed2/pod.env with POD_IP=... POD_PORT=... (written after `runpodctl pod get <id>`).
# never syncs .env (API keys stay on the laptop)
#   ./sync.sh push            -> seed2 code (no runs/samples/ext) to /root/proj/seed2 ; also Arrow's repo (small)
#   ./sync.sh pull            -> /workspace/runs/*.jsonl + logs to seed2/samples/
#   ./sync.sh ssh 'cmd; cmd'  -> run a command string on the pod (single argument, runs in the pod's bash)
cd "$(dirname "$0")"
source ./pod.env
SSH=(ssh -o StrictHostKeyChecking=no -o ConnectTimeout=20 -o ServerAliveInterval=15 -o ServerAliveCountMax=4 -i "$HOME/.runpod/ssh/runpodctl-ssh-key" -p "$POD_PORT")
RS=(rsync -az --no-owner --no-group --timeout=120 -e "${SSH[*]}")
case "$1" in
  push)
    "${RS[@]}" --exclude '.env' --exclude '*.env' --exclude 'pod.env' --exclude 'sync.sh' --exclude 'runs' --exclude 'samples' --exclude 'ext' --exclude '__pycache__' ./ "root@$POD_IP:/root/proj/seed2/"
    "${SSH[@]}" "root@$POD_IP" "mkdir -p /root/proj/seed2/ext/character-reward-hacking"
    "${RS[@]}" --exclude '.git' --exclude '.env' ./ext/character-reward-hacking/src ./ext/character-reward-hacking/scripts "root@$POD_IP:/root/proj/seed2/ext/character-reward-hacking/" ;;
  pull)
    mkdir -p samples
    "${RS[@]}" --include '*.jsonl' --include '*.log' --include '*.txt' --include '*.json' --include '*.md' --exclude '*' "root@$POD_IP:/workspace/runs/" ./samples/ ;;
  ssh)
    shift; "${SSH[@]}" "root@$POD_IP" "$*" ;;
  *) echo "usage: sync.sh push|pull|ssh" ;;
esac
