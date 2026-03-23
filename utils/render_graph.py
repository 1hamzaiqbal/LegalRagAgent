"""Render the current LangGraph workflow to graph.png."""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from main import build_graph

def main() -> None:
    app = build_graph()
    png_bytes = app.get_graph().draw_mermaid_png()
    out_path = Path("graph.png")
    out_path.write_bytes(png_bytes)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
