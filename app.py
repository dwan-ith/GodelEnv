"""Convenience entrypoint for local runs and HF-style tooling."""

from server.app import app, main


__all__ = ["app", "main"]


if __name__ == "__main__":
    main()
