import json
from pathlib import Path
from typing import Any, Dict, List, Optional

BASE_DIR = Path(__file__).resolve().parent
SKILLS_DIR = BASE_DIR / ".agents" / "skills"
SKILLS_LOCK_PATH = BASE_DIR / "skills-lock.json"

# ------------------------------------------------------------------------------
# Connectors Catalog (Empty - Clean Marketplace)
# ------------------------------------------------------------------------------
AVAILABLE_CONNECTORS: List[Dict[str, Any]] = []


def register_connector(connector: Dict[str, Any]) -> None:
    """Register or update a connector definition."""
    existing_idx = next(
        (i for i, c in enumerate(AVAILABLE_CONNECTORS) if c["id"] == connector["id"]),
        None,
    )
    if existing_idx is not None:
        AVAILABLE_CONNECTORS[existing_idx] = connector
    else:
        AVAILABLE_CONNECTORS.append(connector)


def clear_all_connectors() -> None:
    """Clear all registered connectors."""
    AVAILABLE_CONNECTORS.clear()


# ------------------------------------------------------------------------------
# Skills Catalog (Empty - Clean Marketplace)
# ------------------------------------------------------------------------------
CUSTOM_SKILLS: List[Dict[str, Any]] = []


def register_custom_skill(skill: Dict[str, Any]) -> None:
    """Register or update a custom skill definition."""
    existing_idx = next(
        (i for i, s in enumerate(CUSTOM_SKILLS) if s["id"] == skill["id"]),
        None,
    )
    if existing_idx is not None:
        CUSTOM_SKILLS[existing_idx] = skill
    else:
        CUSTOM_SKILLS.append(skill)


def clear_all_skills() -> None:
    """Clear all registered skills."""
    CUSTOM_SKILLS.clear()


# ------------------------------------------------------------------------------
# Skills Lockfile Helpers
# ------------------------------------------------------------------------------
def get_installed_skill_ids(
    chat_store: Any = None, user_id: Optional[str] = None
) -> set[str]:
    if chat_store is not None and user_id is not None:
        raw = chat_store.get_setting("marketplace_installed_skills", user_id=user_id)
        if not raw:
            return set()
        try:
            items = json.loads(raw)
            return set(items) if isinstance(items, list) else set()
        except Exception:
            return set()
    if not SKILLS_LOCK_PATH.is_file():
        return set()
    try:
        data = json.loads(SKILLS_LOCK_PATH.read_text(encoding="utf-8"))
        return set(data.get("skills", {}).keys())
    except Exception:
        return set()


def toggle_skill_installed(
    skill_id: str,
    install: bool,
    chat_store: Any = None,
    user_id: Optional[str] = None,
) -> bool:
    if chat_store is not None and user_id is not None:
        installed = get_installed_skill_ids(chat_store=chat_store, user_id=user_id)
        if install:
            installed.add(skill_id)
        else:
            installed.discard(skill_id)
        chat_store.set_setting(
            "marketplace_installed_skills",
            json.dumps(sorted(installed)),
            user_id=user_id,
        )
        return True

    try:
        if SKILLS_LOCK_PATH.is_file():
            data = json.loads(SKILLS_LOCK_PATH.read_text(encoding="utf-8"))
        else:
            data = {"version": 1, "skills": {}}
    except Exception:
        data = {"version": 1, "skills": {}}

    skills_dict = data.setdefault("skills", {})
    if install:
        skills_dict[skill_id] = {
            "source": "Orbit/Marketplace",
            "sourceType": "custom",
            "installed_at": "now",
        }
    else:
        skills_dict.pop(skill_id, None)

    SKILLS_LOCK_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return True


def list_available_skills(
    force_refresh: bool = False,
    chat_store: Any = None,
    user_id: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """List available skills in the marketplace (starts empty)."""
    installed_ids = get_installed_skill_ids(chat_store=chat_store, user_id=user_id)
    return [{**s, "is_installed": s["id"] in installed_ids} for s in CUSTOM_SKILLS]


def get_skill_readme_content(skill_id: str) -> Optional[str]:
    """Retrieve readme content for a skill."""
    custom = next((s for s in CUSTOM_SKILLS if s["id"] == skill_id), None)
    if custom and custom.get("readme"):
        return custom["readme"]
    skill_file = SKILLS_DIR / skill_id / "SKILL.md"
    if skill_file.is_file():
        try:
            return skill_file.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            return None
    return None


# ------------------------------------------------------------------------------
# Connectors Status & Persistence
# ------------------------------------------------------------------------------
def _mask_secret(value: str) -> str:
    if not value or len(value) <= 6:
        return "••••••"
    return f"{value[:3]}••••••{value[-3:]}"


def list_connectors_with_status(
    chat_store: Any, user_id: Optional[str] = None
) -> List[Dict[str, Any]]:
    connector_settings = chat_store.get_settings_by_prefix("connector_", user_id=user_id)
    results = []

    for conn in AVAILABLE_CONNECTORS:
        conn_id = conn["id"]
        enabled_key = f"connector_{conn_id}_enabled"
        config_key = f"connector_{conn_id}_config"

        is_enabled = connector_settings.get(enabled_key) == "1"
        raw_config = connector_settings.get(config_key, "{}")
        try:
            config_dict = json.loads(raw_config) if raw_config else {}
        except Exception:
            config_dict = {}

        # Check required fields
        required_keys = [f["key"] for f in conn.get("configFields", []) if f.get("required")]
        all_required_set = all(bool(config_dict.get(k)) for k in required_keys)
        is_configured = (not required_keys) or all_required_set

        # Mask password fields
        masked_config: dict[str, Any] = {}
        for f in conn.get("configFields", []):
            val = config_dict.get(f["key"])
            if val is not None:
                if f.get("type") == "password":
                    masked_config[f["key"]] = _mask_secret(str(val))
                else:
                    masked_config[f["key"]] = val

        status = "active" if (is_enabled and is_configured) else ("needs_config" if is_enabled else "available")

        results.append({
            **conn,
            "type": "connector",
            "is_enabled": is_enabled,
            "is_configured": is_configured,
            "status": status,
            "config": masked_config,
        })

    return results


def save_connector_configuration(
    chat_store: Any,
    connector_id: str,
    enabled: bool,
    config: Dict[str, Any],
    user_id: Optional[str] = None,
) -> Dict[str, Any]:
    target = next((c for c in AVAILABLE_CONNECTORS if c["id"] == connector_id), None)
    if not target:
        raise KeyError(f"Unknown connector {connector_id}")

    config_key = f"connector_{connector_id}_config"
    enabled_key = f"connector_{connector_id}_enabled"

    existing_raw = chat_store.get_setting(config_key, "{}", user_id=user_id)
    try:
        existing_config = json.loads(existing_raw) if existing_raw else {}
    except Exception:
        existing_config = {}

    merged_config = dict(existing_config)
    for k, v in config.items():
        if isinstance(v, str) and "••••" in v:
            continue
        if v is not None:
            merged_config[k] = v

    chat_store.set_setting(config_key, json.dumps(merged_config, ensure_ascii=False), user_id=user_id)
    chat_store.set_setting(enabled_key, "1" if enabled else "0", user_id=user_id)

    return {
        "connector_id": connector_id,
        "enabled": enabled,
        "is_configured": all(
            bool(merged_config.get(f["key"]))
            for f in target.get("configFields", [])
            if f.get("required")
        ),
    }


def toggle_connector_enabled(
    chat_store: Any,
    connector_id: str,
    enabled: bool,
    user_id: Optional[str] = None,
) -> bool:
    target = next((c for c in AVAILABLE_CONNECTORS if c["id"] == connector_id), None)
    if not target:
        raise KeyError(f"Unknown connector {connector_id}")
    enabled_key = f"connector_{connector_id}_enabled"
    chat_store.set_setting(enabled_key, "1" if enabled else "0", user_id=user_id)
    return True
