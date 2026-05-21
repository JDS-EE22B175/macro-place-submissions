import argparse
import subprocess
import concurrent.futures
import sys
import os

IBM_BENCHMARKS = [
    "ibm01", "ibm02", "ibm03", "ibm04", "ibm06", "ibm07", "ibm08", "ibm09",
    "ibm10", "ibm11", "ibm12", "ibm13", "ibm14", "ibm15", "ibm16", "ibm17", "ibm18"
]

NG45_BENCHMARKS = ["ariane133", "ariane136", "mempool_tile", "nvdla"]

def run_bench(placer, bench):
    cmd = ["uv", "run", "evaluate", placer, "-b", bench]
    print(f"  [>] Started {bench}...")
    res = subprocess.run(cmd, capture_output=True, text=True)
    
    # Save the full log
    log_path = f"eval_results/{bench}.log"
    with open(log_path, "w") as f:
        f.write(res.stdout)
        f.write(res.stderr)
        
    # Extract the summary line
    for line in res.stdout.split('\n'):
        if "proxy=" in line or "INVALID" in line:
            return f"{bench:>13} | {line.strip()}"
            
    return f"{bench:>13} | Execution failed or timed out. See {log_path}"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parallel Evaluation Wrapper")
    parser.add_argument("placer", help="Path to the placer script")
    parser.add_argument("--all", action="store_true", help="Run IBM benchmarks")
    parser.add_argument("--ng45", action="store_true", help="Run NG45 benchmarks")
    args = parser.parse_args()

    if args.ng45:
        benches = NG45_BENCHMARKS
    elif args.all:
        benches = IBM_BENCHMARKS
    else:
        print("Please specify --all or --ng45")
        sys.exit(1)

    os.makedirs("eval_results", exist_ok=True)
    print(f"Running parallel evaluation on {len(benches)} benchmarks using 8 cores...")
    print("-" * 80)
    
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(run_bench, args.placer, b): b for b in benches}
        for future in concurrent.futures.as_completed(futures):
            bench = futures[future]
            res = future.result()
            print(res)
            results.append(res)
            
    print("-" * 80)
    print("All completed! Full logs are saved in the eval_results/ directory.")
