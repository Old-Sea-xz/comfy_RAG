"""Network defaults for external model and service requests."""
from __future__ import annotations

import os
from collections.abc import MutableMapping


DEFAULT_EXTERNAL_PROXY_URL = "http://127.0.0.1:7890"
_LOCAL_HOSTS = ("127.0.0.1", "localhost", "::1")


def configure_external_proxy(
    environ: MutableMapping[str, str] | None = None,
) -> MutableMapping[str, str]:
    """Route external HTTP(S) traffic through the configured local proxy.

    The local ComfyUI API remains direct through ``NO_PROXY``.  Explicit
    ``COMFYROUTER_LLM_PROXY_URL`` is also filled so the agent client, which
    intentionally ignores ambient proxy variables, uses the same route.
    """
    env = environ if environ is not None else os.environ
    proxy = env.get("COMFYROUTER_EXTERNAL_PROXY_URL", DEFAULT_EXTERNAL_PROXY_URL).strip()
    if not proxy:
        for key in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY",
                    "http_proxy", "https_proxy", "all_proxy"):
            env[key] = ""
        env.setdefault("COMFYROUTER_LLM_PROXY_URL", "")
        return env
    for key in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY",
                "http_proxy", "https_proxy", "all_proxy"):
        env[key] = proxy
    existing = [item.strip() for item in env.get("NO_PROXY", "").split(",")]
    no_proxy = list(dict.fromkeys([*(_LOCAL_HOSTS), *(item for item in existing if item)]))
    env["NO_PROXY"] = ",".join(no_proxy)
    env["no_proxy"] = env["NO_PROXY"]
    # An explicitly empty LLM proxy selects direct transport, as in config.load_env.
    if "COMFYROUTER_LLM_PROXY_URL" not in env:
        env["COMFYROUTER_LLM_PROXY_URL"] = proxy
    env.setdefault("HF_HUB_DISABLE_XET", "1")
    return env
