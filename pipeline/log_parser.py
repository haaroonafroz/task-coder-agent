"""
Parse llama.cpp server logs for speculative decoding acceptance rates.
MTP logs typically include lines like:
  - "speculative: accepted N / drafted M tokens (XX.XX%)"
  - Or per-request stats in the timing summary
"""
import re
import json
from pathlib import Path
from typing import Optional


def parse_acceptance_rate_from_log(log_path: Path, request_pattern: Optional[str] = None) -> Optional[float]:
    """
    Extract acceptance rate from llama.cpp server log.
    
    llama.cpp with MTP typically logs acceptance stats after each request:
    - Look for patterns like "speculative: accepted X / Y tokens (Z%)"
    - Or parse from "print_timing" output
    
    Returns: Acceptance rate as float (0.0-1.0) or None if not found
    """
    if not log_path.exists():
        return None
    
    content = log_path.read_text()
    
    # Pattern 1: Direct speculative acceptance line
    # Example: "speculative: accepted 12 / 16 tokens (75.00%)"
    pattern1 = r"draft acceptance rate = ([\d.]+) \(\s*(\d+) accepted / \s*(\d+) generated\)"
    matches1 = re.findall(pattern1, content)
    if matches1:
        # Average all matches or use the last one
        rates = [float(rate) for rate, _, _ in matches1]
        return sum(rates) / len(rates)
    
    # Pattern 2: llama.cpp timing output with draft stats
    # Example: "draft_tokens = 16, accepted_tokens = 12"
    pattern2 = r"draft_tokens = (\d+), accepted_tokens = (\d+)"
    matches2 = re.findall(pattern2, content)
    if matches2:
        total_draft = sum(int(d) for d, _ in matches2)
        total_accept = sum(int(a) for _, a in matches2)
        return total_accept / total_draft if total_draft > 0 else None
    
    # Pattern 3: print_timing speculative section
    # Look for speculative decoding section in timing output
    pattern3 = r"speculative.*?:.*?(\d+)/(\d+)"
    matches3 = re.findall(pattern3, content, re.DOTALL)
    if matches3:
        total_accept = sum(int(a) for a, _ in matches3)
        total_draft = sum(int(d) for _, d in matches3)
        return total_accept / total_draft if total_draft > 0 else None
    
    return None
def update_results_with_acceptance_rates(results_path: Path, log_path: Path, mode: str = "speculative") -> None:
    """
    Post-process results.csv to add acceptance rates from server logs.
    llama.cpp logs a GLOBAL acceptance rate, not per-agent.
    """
    import csv
    
    # Parse global acceptance rate from log
    global_rate = parse_acceptance_rate_from_log(log_path)
    
    if global_rate is None:
        print(f"[WARNING] Could not parse acceptance rate from {log_path}")
        return
    
    print(f"[INFO] Parsed global acceptance rate: {global_rate:.2%}")
    
    # Update CSV - apply global rate to all agents
    rows = []
    with open(results_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        
        # Add acceptance rate columns if they don't exist
        for agent in ["planner", "generator", "refiner"]:
            col = f"acceptance_rate_{agent}"
            if col not in fieldnames:
                fieldnames.append(col)
        # Also add a global column for clarity
        if "acceptance_rate_global" not in fieldnames:
            fieldnames.append("acceptance_rate_global")
        
        for row in reader:
            if row.get("mode") == mode:
                # Apply the SAME global rate to all agents
                # (llama.cpp doesn't track per-agent acceptance)
                row["acceptance_rate_planner"] = f"{global_rate:.4f}"
                row["acceptance_rate_generator"] = f"{global_rate:.4f}"
                row["acceptance_rate_refiner"] = f"{global_rate:.4f}"
                row["acceptance_rate_global"] = f"{global_rate:.4f}"
            rows.append(row)
    
    # Write back
    with open(results_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    
    print(f"[OK] Updated {results_path} with global acceptance rate {global_rate:.2%}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: python pipeline/log_parser.py <results.csv> <server.log>")
        sys.exit(1)
    
    results_path = Path(sys.argv[1])
    log_path = Path(sys.argv[2])
    update_results_with_acceptance_rates(results_path, log_path)