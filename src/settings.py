"""Settings persistence: JSON load/save plus migration from the old txt files."""
import json

import logging

logger = logging.getLogger(__name__)

from runtime import ROOT, SETTINGS_FILE


def load_settings():
    if SETTINGS_FILE.exists():
        try:
            data = json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
        except (OSError, ValueError) as exc:
            logger.warning(f"Could not read {SETTINGS_FILE}: {exc}")
            return {}
    migrated = {}
    old_prompt = ROOT / "system_prompt.txt"
    if old_prompt.exists():
        migrated["system_prompt"] = old_prompt.read_text(encoding="utf-8").strip()
    old_name = ROOT / "avatar_name.txt"
    if old_name.exists():
        migrated["avatar_name"] = old_name.read_text(encoding="utf-8").strip()
    if migrated:
        logger.info("Migrated settings from system_prompt/avatar_name txt files.")
    return migrated


def write_settings(data):
    try:
        SETTINGS_FILE.write_text(
            json.dumps(data, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    except OSError as exc:
        logger.error(f"Could not save {SETTINGS_FILE}: {exc}")
