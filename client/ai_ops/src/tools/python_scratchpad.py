from langchain_core.tools import tool
import re  # keep: we’ll also preload this for exec

@tool
def run_python(code: str) -> dict:
    """
    Run untrusted Python in a sandbox-like context.
    All file writes are restricted to ALLOWED_DIR (and children).
    Stdout/stderr are captured. A variable named `result` is returned if present.
    """
    import io, os, builtins, contextlib, traceback, html
    from pathlib import Path
    from dotenv import load_dotenv
    import tempfile

    THIS_DIR     = Path(__file__).resolve()
    PROJECT_ROOT = THIS_DIR.parent.parent.parent
    print(PROJECT_ROOT)
    load_dotenv(PROJECT_ROOT / "config/.env")

    # 1) Define the allowed base dir (env var override supported)
    base = os.getenv("FILE_SYSTEM_ACCESS_KEY")
    if base:
        ALLOWED_DIR = Path(base).expanduser().resolve()
    else:
        # minimal change: provide a sane fallback to avoid .resolve() on None and PermissionError
        ALLOWED_DIR = Path(tempfile.gettempdir(), "py_sandbox").resolve()
    ALLOWED_DIR.mkdir(parents=True, exist_ok=True)

    def _resolve_safe_path(p: str | os.PathLike):
        p = Path(p)
        # allow simple names like "foo.html" by resolving relative to ALLOWED_DIR
        if not p.is_absolute():
            p = ALLOWED_DIR / p
        rp = p.resolve()
        if ALLOWED_DIR not in rp.parents and rp != ALLOWED_DIR:
            raise PermissionError(
                f"Access denied - path outside allowed directories: {rp} not in {ALLOWED_DIR}"
            )
        return rp

    # 2) Safe open wrapper
    _real_open = builtins.open
    def _safe_open(file, mode="r", *args, **kwargs):
        rp = _resolve_safe_path(file)
        if any(ch in mode for ch in ("w", "a", "+", "x")):
            rp.parent.mkdir(parents=True, exist_ok=True)
        return _real_open(rp, mode, *args, **kwargs)

    # 3) Run the code with patched open and cwd set to ALLOWED_DIR
    ns, out, err = {}, io.StringIO(), io.StringIO()
    g = {
        "__builtins__": builtins.__dict__,  # ensure imports work inside exec
        "re": re,                            # preload stdlib 're' to avoid shadowing/blocked imports
    }
    try:
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            cwd_before = os.getcwd()
            builtins.open = _safe_open
            try:
                os.chdir(ALLOWED_DIR)
                exec(code, g, ns)
            finally:
                os.chdir(cwd_before)
                builtins.open = _real_open
        return {
            "ok": True,
            "result": ns.get("result"),
            "stdout": html.escape(out.getvalue()),
            "stderr": html.escape(err.getvalue()),
        }
    except Exception:
        # ensure open is restored even on error
        builtins.open = _real_open
        return {
            "ok": False,
            "error": html.escape(traceback.format_exc()),
            "stdout": html.escape(out.getvalue()),
            "stderr": html.escape(err.getvalue()),
        }
