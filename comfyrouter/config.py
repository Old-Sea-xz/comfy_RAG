"""路径配置。data/ 与本包同级，自包含，不依赖原挖掘产物目录。"""
from __future__ import annotations
import os
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / 'data'
NODE_REGISTRY = Path(os.environ.get('COMFYROUTER_NODE_REGISTRY', str(DATA / 'comfyui_object_info.json')))
AGENT_SKILLS = Path(os.getenv('COMFYROUTER_AGENT_CONTRACTS', str(DATA / 'agent_skills.json')))
EVOLUTION_DIR = Path(os.environ.get('COMFYROUTER_EVOLUTION_DIR', str(DATA / 'evolution')))

def base_only_ablation() -> bool:
    """Opt-in experiment; production defaults are unchanged."""
    return os.getenv('COMFYROUTER_BASE_ONLY_ABLATION', '').casefold() in {'1', 'true', 'yes', 'on'}
ENV_FILE = ROOT / '.env'

def evaluation_model(fallback: str='') -> str:
    """Resolve the independent evaluator model, preserving legacy defaults."""
    return os.getenv('COMFYROUTER_LLM_EVALUATION_MODEL', '').strip() or fallback or os.getenv('COMFYROUTER_LLM_ITERATION_MODEL', '').strip() or os.getenv('COMFYROUTER_LLM_MODEL', '').strip()

def evaluation_api_key(fallback: str='') -> str:
    """Use the evaluator credential when set, otherwise the primary credential."""
    return os.getenv('COMFYROUTER_LLM_EVALUATION_API_KEY', '').strip() or fallback or os.getenv('COMFYROUTER_LLM_API_KEY', '')

def load_env(path: Path | None=None) -> None:
    """Load simple KEY=VALUE settings without requiring python-dotenv.

    Existing nonempty process variables win over the local file. An explicitly
    empty LLM proxy selects direct transport and must also be preserved. Other
    empty values retain the existing file fallback. The file is intended for
    local development and must never be committed with a real secret.
    """
    env_path = path or ENV_FILE
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding='utf-8').splitlines():
        line = raw_line.strip()
        if not line or line.startswith('#') or '=' not in line:
            continue
        key, value = line.split('=', 1)
        key, value = (key.strip(), value.strip())
        if not key:
            continue
        if len(value) >= 2 and value[0] == value[-1] and (value[0] in {"'", '"'}):
            value = value[1:-1]
        explicit_direct_proxy = key in {'COMFYROUTER_LLM_PROXY_URL', 'COMFYROUTER_JEV_PROXY_URL'} and key in os.environ
        if not os.environ.get(key) and (not explicit_direct_proxy):
            os.environ[key] = value
load_env()
COMFYUI_ROOT = Path(os.getenv('COMFYROUTER_COMFYUI_ROOT', str(ROOT.parent / 'ComfyUI')))
COMFYBENCH_ROOT = Path(os.getenv('COMFYROUTER_COMFYBENCH_ROOT', str(ROOT.parent / 'ComfyBench')))
