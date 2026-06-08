"""scripts/promote.py — promote MLflow Registry aliases with an audit log.

YOUR TASK (see tasks/task2.md): implement the four subcommand functions.
The argparse scaffolding below is wired so each cmd_* receives an `args`
namespace already parsed. See `_build_parser` for what's on `args` per
subcommand, and tasks/task2.md "Behavioral specs" for what each function
must do.

Versions are identified by their `config_id` tag (e.g., "v6"), NOT by
MLflow's integer version numbers. Resolution must be unique — if the
config_id matches zero or multiple registered versions, the CLI errors
out and forces the operator to disambiguate via the MLflow UI.

Successful `set` and `rollback` operations append a JSON event to
LOG_FILE (promotion-log.jsonl at repo root). `rollback` consults the
log to find the previous alias target.

Subcommands:
  set <alias> <config_id>   move alias, append `set` event to the log
  show <alias>              print current target + tags + key metrics
  list                      print all aliases on the registered model
  rollback <alias>          move alias back per the audit log, append
                            `rollback` event
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import mlflow
import mlflow.exceptions
from mlflow.tracking import MlflowClient

REGISTERED_MODEL_NAME = "travel-assistant"
LOG_FILE = Path(__file__).resolve().parent.parent / "promotion-log.jsonl"

# Load .env and point mlflow at the tracking server
_repo_root = Path(__file__).resolve().parent.parent
try:
    from dotenv import load_dotenv
    load_dotenv(_repo_root / ".env")
except ImportError:
    pass

import os
_tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
mlflow.set_tracking_uri(_tracking_uri)

client = MlflowClient()


def _find_version_by_config_id(config_id: str, model_name: str):
    """Return the ModelVersion whose config_id tag matches. Handles 0 or N>1 matches."""
    versions = client.search_model_versions(
        f"name = '{model_name}' AND tags.config_id = '{config_id}'"
    )
    if len(versions) == 0:
        print(f"error: no version found with config_id={config_id}")
        sys.exit(1)
    if len(versions) > 1:
        version_nums = sorted(int(v.version) for v in versions)
        print(
            f"warning: multiple versions match config_id={config_id} "
            f"(MLflow versions {version_nums}); using latest ({version_nums[-1]})"
        )
        latest = version_nums[-1]
        versions = [v for v in versions if int(v.version) == latest]
    return versions[0]


def _read_log() -> list[dict]:
    if not LOG_FILE.exists():
        return []
    entries = []
    for line in LOG_FILE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            entries.append(json.loads(line))
    return entries


def _append_log(entry: dict) -> None:
    with LOG_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


def cmd_set(args: argparse.Namespace) -> None:
    """args.alias: str, args.config_id: str. See tasks/task2.md → cmd_set."""
    mv = _find_version_by_config_id(args.config_id, args.name)

    try:
        current_mv = client.get_model_version_by_alias(args.name, args.alias)
        from_config_id = current_mv.tags.get("config_id", "")
    except mlflow.exceptions.RestException:
        from_config_id = ""

    client.set_registered_model_alias(args.name, args.alias, mv.version)

    _append_log({
        "ts": datetime.now(timezone.utc).isoformat(),
        "alias": args.alias,
        "from": from_config_id,
        "to": args.config_id,
        "op": "set",
    })

    from_label = from_config_id if from_config_id else "(unset)"
    print(f"{args.alias}: {from_label} → {args.config_id}")


def cmd_show(args: argparse.Namespace) -> None:
    """args.alias: str. See tasks/task2.md → cmd_show."""
    try:
        mv = client.get_model_version_by_alias(args.name, args.alias)
    except mlflow.exceptions.RestException:
        print(f"error: alias '{args.alias}' is not set")
        sys.exit(1)

    config_id = mv.tags.get("config_id", "?")
    model = mv.tags.get("model", "?")
    guardrail_type = mv.tags.get("guardrail_type", "?")

    run = client.get_run(mv.run_id)
    m = run.data.metrics

    print(f"{args.name} @ {args.alias}")
    print(f"  config_id: {config_id}")
    print(f"  model: {model}")
    print(f"  guardrail_type: {guardrail_type}")
    print(f"  accuracy_overall: {m.get('accuracy_overall', float('nan')):.2f}")
    print(f"  verdict_rate_leaked: {m.get('verdict_rate_leaked', 0.0):.2f}")
    print(f"  total_cost_usd: ${m.get('total_cost_usd', 0.0):.4f}")


def cmd_list(args: argparse.Namespace) -> None:
    """No args. See tasks/task2.md → cmd_list."""
    try:
        rm = client.get_registered_model(args.name)
    except mlflow.exceptions.RestException:
        print("no aliases set")
        return

    aliases = rm.aliases  # {alias_name: version_number_str}
    if not aliases:
        print("no aliases set")
        return

    for alias, version_str in aliases.items():
        mv = client.get_model_version(args.name, version_str)
        config_id = mv.tags.get("config_id", "?")
        print(f"{alias} -> {config_id}")


def cmd_rollback(args: argparse.Namespace) -> None:
    """args.alias: str. See tasks/task2.md → cmd_rollback."""
    try:
        current_mv = client.get_model_version_by_alias(args.name, args.alias)
        current_config_id = current_mv.tags.get("config_id", "")
    except mlflow.exceptions.RestException:
        print("nothing to roll back")
        return

    log = _read_log()
    relevant = [e for e in log if e["alias"] == args.alias]

    if not relevant:
        print(f"no promotion history for alias {args.alias}")
        return

    last_entry = relevant[-1]

    if last_entry["op"] == "rollback":
        print(f"{args.alias} was just rolled back; no further history to walk back to")
        return

    if last_entry["op"] == "set" and last_entry["from"] == "":
        print(f"{args.alias} has no previous target (first promotion ever)")
        return

    prev_config_id = last_entry["from"]
    mv = _find_version_by_config_id(prev_config_id, args.name)
    client.set_registered_model_alias(args.name, args.alias, mv.version)

    _append_log({
        "ts": datetime.now(timezone.utc).isoformat(),
        "alias": args.alias,
        "from": current_config_id,
        "to": prev_config_id,
        "op": "rollback",
    })

    print(f"{args.alias}: {current_config_id} → {prev_config_id} (rolled back)")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--name",
        default=REGISTERED_MODEL_NAME,
        help=f"Registered model name (default: {REGISTERED_MODEL_NAME})",
    )

    sub = parser.add_subparsers(dest="cmd", required=True)

    p_set = sub.add_parser(
        "set", help="Move an alias to a version (by config_id), append a set event"
    )
    p_set.add_argument("alias", help="Alias to assign (e.g., 'production')")
    p_set.add_argument(
        "config_id",
        help="Config identifier (e.g., 'v6') — resolved via the config_id tag on registered versions",
    )
    p_set.set_defaults(func=cmd_set)

    p_show = sub.add_parser("show", help="Show which version an alias points at")
    p_show.add_argument("alias")
    p_show.set_defaults(func=cmd_show)

    p_list = sub.add_parser("list", help="List all aliases on the registered model")
    p_list.set_defaults(func=cmd_list)

    p_rollback = sub.add_parser(
        "rollback",
        help="Move an alias back to its previous target per the audit log",
    )
    p_rollback.add_argument("alias")
    p_rollback.set_defaults(func=cmd_rollback)

    return parser


def main() -> None:
    args = _build_parser().parse_args()
    try:
        args.func(args)
    except NotImplementedError as exc:
        print(f"NOT IMPLEMENTED: {exc}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
