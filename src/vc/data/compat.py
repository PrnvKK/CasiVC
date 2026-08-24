# data/compat.py
"""
Cross-Platform Compatibility & Monkeypatches for SpeechBrain / HuggingFace
========================================================================
Resolves:
  1. Windows Symlink Privilege (WinError 1314): Safely falls back to file copy.
  2. Torchaudio audio backend deprecation.
  3. HuggingFace Hub token and dummy file workarounds.
"""

import os
import sys
import shutil
import warnings
from pathlib import Path

# --- 1. Fix Torchaudio Backends ---
try:
    import torchaudio
    if not hasattr(torchaudio, "list_audio_backends"):
        torchaudio.list_audio_backends = lambda: ["soundfile"]
except ImportError:
    pass

# --- 2. Fix Hugging Face Token deprecation & missing dummy custom.py ---
try:
    import huggingface_hub
    _orig_hf_download = huggingface_hub.hf_hub_download

    def _patched_hf_download(*args, **kwargs):
        if "use_auth_token" in kwargs:
            kwargs["token"] = kwargs.pop("use_auth_token")
        filename = kwargs.get("filename")
        if filename is None and len(args) > 1:
            filename = args[1]
        try:
            return _orig_hf_download(*args, **kwargs)
        except Exception as e:
            if filename == "custom.py" and ("404" in str(e) or "Not Found" in str(e)):
                cache_dir = kwargs.get("cache_dir", "cache/speechbrain_dummy")
                dummy_path = os.path.join(cache_dir, "custom.py")
                os.makedirs(os.path.dirname(dummy_path), exist_ok=True)
                with open(dummy_path, "w", encoding="utf-8") as f:
                    f.write("")
                return dummy_path
            raise e

    huggingface_hub.hf_hub_download = _patched_hf_download
except Exception:
    pass

# --- 3. Fix Windows Symlink Privilege (WinError 1314) ---
# SpeechBrain's link_with_strategy tries to create symlinks by default.
# Non-admin Windows accounts raise OSError [WinError 1314].
# We safely patch link_with_strategy and Path.symlink_to to fall back to copy2.
try:
    import speechbrain.utils.fetching as sb_fetch

    def _safe_link_with_strategy(src, dst, local_strategy=None):
        src_p = Path(src)
        dst_p = Path(dst)
        dst_p.parent.mkdir(parents=True, exist_ok=True)

        if dst_p.exists() or dst_p.is_symlink():
            try:
                if dst_p.is_dir() and not dst_p.is_symlink():
                    shutil.rmtree(dst_p)
                else:
                    dst_p.unlink()
            except Exception:
                pass

        try:
            dst_p.symlink_to(src_p)
        except OSError:
            # Fallback to copy on Windows without symlink privileges
            if src_p.is_dir():
                shutil.copytree(src_p, dst_p, dirs_exist_ok=True)
            else:
                shutil.copy2(src_p, dst_p)

    sb_fetch.link_with_strategy = _safe_link_with_strategy
except Exception:
    pass

# Also patch Path.symlink_to and os.symlink as secondary defense
_orig_symlink = os.symlink
def _safe_os_symlink(src, dst, *args, **kwargs):
    try:
        return _orig_symlink(src, dst, *args, **kwargs)
    except OSError:
        if os.path.isdir(src):
            shutil.copytree(src, dst, dirs_exist_ok=True)
        else:
            shutil.copy2(src, dst)
os.symlink = _safe_os_symlink

_orig_path_symlink_to = Path.symlink_to
def _safe_path_symlink_to(self, target, target_is_directory=False):
    try:
        return _orig_path_symlink_to(self, target, target_is_directory=target_is_directory)
    except OSError:
        if os.path.isdir(target):
            shutil.copytree(target, self, dirs_exist_ok=True)
        else:
            shutil.copy2(target, self)
Path.symlink_to = _safe_path_symlink_to
