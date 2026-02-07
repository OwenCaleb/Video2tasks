"""Server CLI entrypoint."""

import click
from pathlib import Path
from ..config import Config
from video2tasks.server import run_server

@click.command()
@click.option(
    "--config", "-c",
    type=click.Path(exists=True, path_type=Path),
    help="Path to configuration file"
)
@click.option(
    "--mode",
    type=click.Choice(["segment", "vqa", "cot"], case_sensitive=False),
    default=None,
    help="Override run.task_type from config"
)
def main(config: Path, mode: str | None) -> None:
    """Start the Video2Tasks server."""
    if config:
        cfg = Config.from_yaml(config)
    else:
        # Try default config.yaml
        default = Path("config.yaml")
        if default.exists():
            cfg = Config.from_yaml(default)
        else:
            raise click.UsageError(
                "No configuration file specified and config.yaml not found.\n"
                "Use --config to specify a config file or copy config.example.yaml to config.yaml"
            )

    if mode:
        cfg.run.task_type = mode.lower()

    if cfg.run.task_type == "vqa":
        from ..vqa.server_app import run_vqa_server

        run_vqa_server(cfg)
    elif cfg.run.task_type == "cot":
        from ..cot.server_app import run_cot_server

        run_cot_server(cfg)
    else:
        run_server(cfg)


if __name__ == "__main__":
    main()