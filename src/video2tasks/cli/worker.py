"""Worker CLI entrypoint."""

import click
from pathlib import Path
from ..config import Config
from ..worker.runner import run_worker


@click.command()
@click.option(
    "--config", "-c",
    type=click.Path(exists=True, path_type=Path),
    help="Path to configuration file"
)
@click.option(
    "--mode",
    type=click.Choice(["segment", "vqa"], case_sensitive=False),
    default=None,
    help="Override run.task_type from config"
)
def main(config: Path, mode: str | None) -> None:
    """Start the Video2Tasks worker."""
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
        from ..vqa.worker_runner import run_vqa_worker

        run_vqa_worker(cfg)
    else:
        run_worker(cfg)


if __name__ == "__main__":
    main()