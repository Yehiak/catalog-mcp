"""Command line: `catalog-mcp [serve|build|stats]`.

With no subcommand it serves over stdio, which is what MCP desktop clients expect when
they launch `uvx catalog-mcp`.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys

from .config import ALL_CATEGORIES, DEFAULT_CATEGORIES, DEFAULT_MODEL, paths


def _add_data_dir(p: argparse.ArgumentParser) -> None:
    p.add_argument("--data-dir", default=None,
                   help="Catalog directory (default: $CATALOG_MCP_HOME or the user data dir).")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="catalog-mcp", description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command")

    s = sub.add_parser("serve", help="Run the MCP server (default).")
    _add_data_dir(s)
    s.add_argument("--transport", choices=["stdio", "http"], default="stdio")
    s.add_argument("--host", default="127.0.0.1", help="HTTP bind address.")
    s.add_argument("--port", type=int, default=8000, help="HTTP port.")
    s.add_argument("--token", default=None,
                   help="Bearer token for HTTP (default: $CATALOG_MCP_TOKEN). Prefer the env var.")
    s.add_argument("--no-auth", action="store_true",
                   help="Serve HTTP without a token. Only allowed on a loopback address.")
    s.add_argument("--no-warm-up", action="store_true", help="Don't preload the embedding model at start.")

    b = sub.add_parser("build", help="Download the dataset and build the DuckDB + FAISS catalog.")
    _add_data_dir(b)
    b.add_argument("--categories", nargs="+", default=list(DEFAULT_CATEGORIES), choices=ALL_CATEGORIES,
                   metavar="CATEGORY", help=f"Dataset categories (default: {' '.join(DEFAULT_CATEGORIES)}).")
    b.add_argument("--model", default=DEFAULT_MODEL, help="sentence-transformers model id.")
    b.add_argument("--device", default=None, help="cuda / cpu / mps (default: auto).")
    b.add_argument("--batch-size", type=int, default=512)

    st = sub.add_parser("stats", help="Print what the built catalog contains.")
    _add_data_dir(st)

    sub.add_parser("token", help="Print a new random token for HTTP auth.")
    return parser


def serve(args: argparse.Namespace) -> None:
    from .server import create_server

    # stdout belongs to the MCP protocol on stdio: all logging goes to stderr.
    logging.basicConfig(level=logging.WARNING, stream=sys.stderr)
    token = None
    if args.transport == "http":
        token = args.token or os.environ.get("CATALOG_MCP_TOKEN")
        loopback = args.host in ("127.0.0.1", "localhost", "::1")
        if not token and not args.no_auth:
            sys.exit("HTTP needs a token: set CATALOG_MCP_TOKEN (see `catalog-mcp token`), or pass --no-auth on localhost.")
        if args.no_auth and not loopback:
            sys.exit("--no-auth is only allowed when binding to a loopback address.")

    server = create_server(args.data_dir, warm_up=not args.no_warm_up)
    if args.transport == "stdio":
        server.run("stdio")
        return

    import uvicorn

    from .auth import BearerTokenMiddleware

    app = server.streamable_http_app(streamable_http_path="/mcp", host=args.host)
    if token:
        app = BearerTokenMiddleware(app, token)
    print(f"catalog-mcp: Streamable HTTP on http://{args.host}:{args.port}/mcp "
          f"({'bearer token required' if token else 'no auth'})", file=sys.stderr)
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


def build_cmd(args: argparse.Namespace) -> None:
    from .embed import SentenceTransformerEmbedder
    from .ingest import build

    p = paths(args.data_dir)
    print(f"building catalog in {p.root}", file=sys.stderr)
    info = build(p, args.categories, embedder=SentenceTransformerEmbedder(args.model, device=args.device),
                 batch_size=args.batch_size, log=lambda m: print(m, file=sys.stderr))
    print(json.dumps(info, indent=2))


def stats_cmd(args: argparse.Namespace) -> None:
    p = paths(args.data_dir)
    if not p.exists():
        sys.exit(f"No catalog in {p.root}. Run `catalog-mcp build` first.")
    print(p.build_info.read_text(encoding="utf-8"))


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] not in {"serve", "build", "stats", "token", "-h", "--help"}:
        argv = ["serve", *argv]  # `catalog-mcp --data-dir X` == `catalog-mcp serve --data-dir X`
    args = build_parser().parse_args(argv)
    if args.command == "serve":
        serve(args)
    elif args.command == "build":
        build_cmd(args)
    elif args.command == "stats":
        stats_cmd(args)
    elif args.command == "token":
        from .auth import generate_token

        print(generate_token())


if __name__ == "__main__":
    main()
