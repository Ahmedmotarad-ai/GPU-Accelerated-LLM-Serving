import flag_gems
print("flag_gems OK")

import vllm_fl
print("vllm_fl OK")

import vllm
print("vllm", vllm.__version__)
print("plugins:", sorted(getattr(vllm.plugins, "load_plugins", lambda: [])() ) if hasattr(vllm.plugins, "load_plugins") else "n/a")