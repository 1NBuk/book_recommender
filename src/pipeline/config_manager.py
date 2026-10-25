
import os
import yaml
from pathlib import Path


class ConfigManager:
    def __init__(self, config_path, project_root=None):
        self.config_path = Path(config_path)
        if project_root is None:
            self.project_root = self.config_path.parent.parent
        else:
            self.project_root = Path(project_root)
        self.config = self._load()
        self._validate()
        self._resolve_paths()

    def _load(self):
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config not found: {self.config_path}")

        with open(self.config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        if config and "extends" in config:
            parent_path = self.config_path.parent / config["extends"]
            parent = ConfigManager(str(parent_path), project_root=self.project_root).config
            config = self._deep_merge(parent, config)
            config.pop("extends", None)

        return config

    def _deep_merge(self, base, override):
        result = dict(base)
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        return result

    def _validate(self):
        required = ["project", "data", "models", "evaluation"]
        for section in required:
            if section not in self.config:
                raise ValueError(f"Missing section in config: {section}")

    def _resolve_paths(self):
        path_keys = ["raw_dir", "processed_dir", "feature_store", "models_dir", "experiments_dir"]
        data = self.config.get("data", {})
        for key in path_keys:
            if key in data and not os.path.isabs(str(data[key])):
                data[key] = str(self.project_root / data[key])

        paths = self.config.get("paths", {})
        for key, value in paths.items():
            if not os.path.isabs(str(value)):
                paths[key] = str(self.project_root / value)

    def get(self, key, default=None):
        keys = key.split(".")
        value = self.config
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        return value

    def set(self, key, value):
        keys = key.split(".")
        target = self.config
        for k in keys[:-1]:
            target = target.setdefault(k, {})
        target[keys[-1]] = value

    def update(self, updates):
        for key, value in updates.items():
            self.set(key, value)

    def save(self, path=None):
        save_path = Path(path) if path else self.config_path
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w", encoding="utf-8") as f:
            yaml.dump(self.config, f, default_flow_style=False, allow_unicode=True)

    def as_dict(self):
        return dict(self.config)

    def __repr__(self):
        return f"ConfigManager({self.config_path})"
