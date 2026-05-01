
from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from pathlib import Path

# local imports 
from agent   import TriageAgent
from logger  import RunLogger
from rag     import Retriever, load_corpus


# defaults (relative to code/) 
DEFAULT_INPUT  = Path(__file__).parent.parent / "support_tickets" / "support_tickets.csv"
DEFAULT_OUTPUT = Path(__file__).parent.parent / "support_tickets" / "output.csv"
DEFAULT_DATA   = Path(__file__).parent.parent / "data"
DEFAULT_LOG    = Path(__file__).parent / "log.txt"

OUTPUT_FIELDS = [
    "issue", "subject", "company",
    "response", "product_area", "status", "request_type", "justification",
]


# CLI 

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Multi-domain support triage agent")
    p.add_argument("--input",  default=str(DEFAULT_INPUT),  help="Path to support_tickets.csv")
    p.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Path to write output.csv")
    p.add_argument("--data",   default=str(DEFAULT_DATA),   help="Path to data/ corpus directory")
    p.add_argument("--log",    default=str(DEFAULT_LOG),    help="Path to log.txt")
    p.add_argument("--limit",  type=int, default=None,      help="Process only first N rows (for testing)")
    return p.parse_args()


# main 

def main():
    args = parse_args()

    # Validate env
    if not os.environ.get("GROQ_API_KEY"):
        print("ERROR: GROQ_API_KEY environment variable is not set.", file=sys.stderr)
        sys.exit(1)

    input_path  = Path(args.input)
    output_path = Path(args.output)
    data_path   = Path(args.data)

    if not input_path.exists():
        print(f"ERROR: Input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)
    if not data_path.exists():
        print(f"ERROR: Data directory not found: {data_path}", file=sys.stderr)
        sys.exit(1)

    # build corpus & retriever 
    print(f"\n{'─'*60}")
    print("  HackerRank Orchestrate — Support Triage Agent")
    print(f"{'─'*60}")
    print(f"  Loading corpus from: {data_path}")
    t_load = time.time()
    chunks = load_corpus(data_path)
    if not chunks:
        print("ERROR: No corpus chunks loaded — check data/ directory.", file=sys.stderr)
        sys.exit(1)
    retriever = Retriever(chunks)
    print(f"  Corpus: {len(chunks)} chunks loaded in {time.time() - t_load:.1f}s")

    # read input CSV 
    with input_path.open(encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        rows   = list(reader)

    if args.limit:
        rows = rows[: args.limit]

    total = len(rows)
    print(f"  Tickets to process: {total}")
    print(f"  Output: {output_path}")
    print(f"  Log:    {args.log}")
    print(f"{'─'*60}\n")

    # setup
    agent  = TriageAgent(retriever)
    logger = RunLogger(args.log)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    out_fh     = output_path.open("w", newline="", encoding="utf-8")
    writer     = csv.DictWriter(out_fh, fieldnames=OUTPUT_FIELDS, extrasaction="ignore")
    writer.writeheader()

    # process 
    for idx, row in enumerate(rows):
        issue   = row.get("issue",   "") or ""
        subject = row.get("subject", "") or ""
        company = row.get("company", "") or ""

        try:
            result = agent.process(issue, subject, company)
        except Exception as exc:
            logger.log_error(idx, str(exc))
            result_row = {
                "issue":         issue,
                "subject":       subject,
                "company":       company,
                "response":      "An unexpected error occurred. Please contact support.",
                "product_area":  "general_support",
                "status":        "escalated",
                "request_type":  "product_issue",
                "justification": f"Agent error: {exc}",
            }
            writer.writerow(result_row)
            out_fh.flush()
            continue

        out_row = {
            "issue":         issue,
            "subject":       subject,
            "company":       company,
            "response":      result.response,
            "product_area":  result.product_area,
            "status":        result.status,
            "request_type":  result.request_type,
            "justification": result.justification,
        }
        writer.writerow(out_row)
        out_fh.flush()

        logger.log_ticket(
            row_idx          = idx,
            issue            = issue,
            subject          = subject,
            company          = company,
            status           = result.status,
            product_area     = result.product_area,
            request_type     = result.request_type,
            response         = result.response,
            justification    = result.justification,
            retrieval_sources = result.sources,
            escalation_rule  = result.escalation_rule,
            elapsed_s        = result.elapsed_s,
        )

    out_fh.close()
    logger.close(total)


if __name__ == "__main__":
    main()