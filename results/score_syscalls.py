#!/usr/bin/env python3
"""
Script to compute addrfilter and seccomp scores based on syscall usage and threat ranking.
Usage:
    python score_syscalls.py --input syscalls.json --ranking syscall-ranking.yaml

The input JSON should be in the form:
{
  "/path/to/bin": {"0": 10, "59": 2, ...},
  ...
}

The ranking YAML should define high-threat and medium-threat syscalls:
syscalls:
  high-threat: [90, 91, 92, ...]
  medium-threat: [16, 83, 84, ...]

Scoring weights:
  - High-threat: 3
  - Medium-threat: 2
  - Other syscalls: 1
"""
import argparse
import json
import sys

try:
    import yaml
except ImportError:
    sys.stderr.write("PyYAML is required. Install via `pip install PyYAML`\n")
    sys.exit(1)


def load_ranking(path):
    """Load syscall ranking YAML and return a dict mapping syscall numbers to custom threat scores."""
    with open(path) as f:
        data = yaml.safe_load(f)
    mapping = {}
    # High-threat syscalls get score 3
    for num in data.get("syscalls", {}).get("high-threat", []):
        mapping[int(num)] = 3
    # Medium-threat syscalls get score 2
    for num in data.get("syscalls", {}).get("medium-threat", []):
        # Don't override high-threat
        mapping.setdefault(int(num), 2)
    return mapping


def load_syscalls(path):
    """Load syscall usage JSON and return a dict of filepath -> set of syscall numbers."""
    with open(path) as f:
        raw = json.load(f)
    result = {}
    for filepath, calls in raw.items():
        syscalls = set()
        for key in calls.keys():
            try:
                syscalls.add(int(key))
            except ValueError:
                continue
        result[filepath] = syscalls
    return result


def compute_addrfilter_score(usage, ranking_map):
    """Compute the addrfilter score: maximum per-file sum of threat scores (default 1)."""
    scores = []
    for syscalls in usage.values():
        score = sum(ranking_map.get(num, 1) for num in syscalls)
        scores.append(score)
    return max(scores) if scores else 0


def compute_seccomp_score(usage, ranking_map):
    """Compute the seccomp score: sum of threat scores over union of all syscalls (default 1)."""
    all_syscalls = set().union(*usage.values()) if usage else set()
    return sum(ranking_map.get(num, 1) for num in all_syscalls)


def main():
    parser = argparse.ArgumentParser(
        description="Compute addrfilter and seccomp scores."
    )
    parser.add_argument(
        "--input", "-i", required=True, help="Path to syscalls JSON file"
    )
    parser.add_argument(
        "--ranking", "-r", required=True, help="Path to syscall-ranking YAML file"
    )
    args = parser.parse_args()

    ranking_map = load_ranking(args.ranking)
    usage = load_syscalls(args.input)

    addrfilter = compute_addrfilter_score(usage, ranking_map)
    seccomp = compute_seccomp_score(usage, ranking_map)

    percentage_reduction = (seccomp - addrfilter) / seccomp

    print(f"addrfilter: {addrfilter}")
    print(f"seccomp: {seccomp}")
    print(f"privilege reduction: {percentage_reduction:.2%}")


if __name__ == "__main__":
    main()
