#!/usr/bin/env bash

find . -name '*.json' | xargs -I {} sh -c 'echo "Scoring: {}" && python3 ../score_syscalls.py -r ../syscall-ranking.yaml -i "{}"'
