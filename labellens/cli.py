"""Command line entry point. Usage: python -m labellens.cli ask "What are the contraindications of metformin?" """
import argparse

from labellens import rag


def main() -> None:
    parser = argparse.ArgumentParser(prog="labellens")
    sub = parser.add_subparsers(dest="command", required=True)
    ask = sub.add_parser("ask", help="answer a question with citations")
    ask.add_argument("question")
    args = parser.parse_args()

    if args.command == "ask":
        result = rag.answer(args.question)
        print(result.text)
        if result.citations:
            print("\nSources:")
            for i, c in enumerate(result.citations, 1):
                print(f"  [{i}] {c.brand_name or c.generic_name} / {c.section} (distance {c.distance:.3f})")
        if result.completion:
            cmp = result.completion
            print(f"\n{cmp.model}: {cmp.prompt_tokens}+{cmp.completion_tokens} tokens, {cmp.latency_ms} ms")


if __name__ == "__main__":
    main()
