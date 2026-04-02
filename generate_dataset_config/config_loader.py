"""Configuration loader for C-PAC LLM Configuration Generator.

Loads settings from config.yaml and provides environment variable overrides.
"""

import os
import yaml
from pathlib import Path
from typing import Any, Dict, Optional


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Load configuration from YAML file.
    
    Args:
        config_path: Path to config.yaml. If None, looks for config.yaml
                    in the same directory as this file.
    
    Returns:
        Dict containing all configuration settings.
    """
    if config_path is None:
        config_path = Path(__file__).parent / "config.yaml"
    else:
        config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # Apply environment variable overrides
    config = _apply_env_overrides(config)
    
    return config


def _apply_env_overrides(config: Dict[str, Any]) -> Dict[str, Any]:
    """Apply environment variable overrides to configuration.
    
    Supported environment variables:
    - OPENAI_API_KEY: Overrides llm_api.openai_api_key
    - OPENAI_BASE_URL: Overrides llm_api.openai_base_url
    - AWS_REGION: Overrides llm_api.aws_region
    - HTTP_PROXY: Overrides llm_api.http_proxy
    - HTTPS_PROXY: Overrides llm_api.https_proxy
    - OLLAMA_HOST: Overrides ollama.host
    - OLLAMA_PORT: Overrides ollama.port
    - LITQUERY_DIR: Overrides rag.litquery_dir
    """
    env_mappings = {
        'OPENAI_API_KEY': ['llm_api', 'openai_api_key'],
        'OPENAI_BASE_URL': ['llm_api', 'openai_base_url'],
        'AWS_REGION': ['llm_api', 'aws_region'],
        'HTTP_PROXY': ['llm_api', 'http_proxy'],
        'HTTPS_PROXY': ['llm_api', 'https_proxy'],
        'OLLAMA_HOST': ['ollama', 'host'],
        'OLLAMA_PORT': ['ollama', 'port'],
        'LITQUERY_DIR': ['rag', 'litquery_dir'],
    }
    
    for env_var, path in env_mappings.items():
        value = os.environ.get(env_var)
        if value is not None:
            _set_nested_value(config, path, value)
    
    return config


def _set_nested_value(data: Dict, path: list, value: Any):
    """Set a value in a nested dictionary based on a path list."""
    current = data
    for key in path[:-1]:
        if key not in current:
            current[key] = {}
        current = current[key]
    current[path[-1]] = value


def get_research_goal(config: Dict[str, Any]) -> str:
    """Get the current research goal text.
    
    If current_research_goal is a key in research_goals, returns that template.
    Otherwise, returns the value directly as a custom goal.
    """
    current_key = config.get('current_research_goal', '')
    research_goals = config.get('research_goals', {})
    
    if current_key in research_goals:
        return research_goals[current_key]
    return current_key


# Global config instance (lazy loaded)
_config_cache: Optional[Dict[str, Any]] = None


def get_config() -> Dict[str, Any]:
    """Get the global configuration instance.
    
    Loads config on first call and caches it.
    """
    global _config_cache
    if _config_cache is None:
        _config_cache = load_config()
    return _config_cache


def reload_config() -> Dict[str, Any]:
    """Reload configuration from file.
    
    Clears the cache and reloads from disk.
    """
    global _config_cache
    _config_cache = load_config()
    return _config_cache


if __name__ == "__main__":
    # Test loading config
    config = load_config()
    print("Configuration loaded successfully!")
    print(f"Backend: {'Ollama' if config['backend']['use_ollama'] else 'AWS Bedrock'}")
    print(f"Models: {config['backend']['ollama_models']}")
    print(f"Research goal preview: {get_research_goal(config)[:100]}...")
