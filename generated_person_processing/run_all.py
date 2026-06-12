#!/usr/bin/env python3
"""Run all person processing scripts."""
import subprocess
import sys

scripts = [
    "process_generated_person_sam1.py",
    "process_generated_person_hqsam.py",
    "process_generated_person_sam3.py",
    "process_generated_person_sam3_text.py",
]

def main():
    for script in scripts:
        print(f"\n{'='*60}")
        print(f"Running {script}")
        print(f"{'='*60}\n")
        result = subprocess.run([sys.executable, script], capture_output=False)
        if result.returncode != 0:
            print(f"Error running {script}")
            return 1
    return 0

if __name__ == "__main__":
    sys.exit(main())
