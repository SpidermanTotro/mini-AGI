"""
Reading the selected YAML settings file.

Both model creation and training read the same settings. MINI_AGI_CONFIG can
select a separate profile; command-line flags still win where they are given.
"""

import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT = os.path.join(ROOT, "config.yaml")
CONFIG_ENV = "MINI_AGI_CONFIG"


def load(path=None):
    import yaml
    path = path or os.environ.get(CONFIG_ENV) or DEFAULT
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return yaml.safe_load(f) or {}


def get(cfg, dotted, fallback=None):
    """get(cfg, 'pool.d_ff') - missing sections return the fallback."""
    cur = cfg
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return fallback
        cur = cur[part]
    return cur
