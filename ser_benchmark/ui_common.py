"""Shared UI utilities used by all pages."""

import streamlit as st


@st.cache_resource
def _get_gpu_info() -> tuple[str, str | None, float | None]:
    """Returns (status, gpu_name, gpu_mem_gb). Cached across reruns."""
    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            return "ok", name, mem
        else:
            return "no_gpu", None, None
    except ImportError:
        return "no_torch", None, None


def show_gpu_sidebar():
    """Display GPU info in the sidebar. Call this on every page."""
    status, name, mem = _get_gpu_info()
    if status == "ok":
        st.sidebar.success(f"\U0001f5a5\ufe0f GPU: {name} ({mem:.1f} GB)")
    elif status == "no_gpu":
        st.sidebar.warning("\u26a0\ufe0f No GPU detected \u2014 inference will be slow")
    else:
        st.sidebar.warning("\u26a0\ufe0f PyTorch not installed")
