"""Experiment wrapper: run Strata's server.py with Inkling thinking NOT forced off.
Stock serve/inkling_dialect.py always renders `reasoning_effort=none` and appends a visible-text opener (no thinking).
This keeps the engine and server identical but renders the chat template with INK_EFFORT (default "low"), no opener.
Usage: python -I serve_think.py <server.py args...>
"""
import os, runpy, sys
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'control')
sys.path.insert(0, ROOT)
from serve import inkling_dialect as d  # noqa: E402
EFF = os.environ.get('INK_EFFORT', 'low')


def template_kwargs(kwargs):
    out = dict(kwargs); out.pop('thinking_mode', None)
    out['reasoning_effort'] = EFF; out['enable_thinking'] = True
    return out


d.template_kwargs = template_kwargs
d.finish_prompt = lambda prompt, **k: prompt
d.normalize_effort = lambda value=None: EFF
d.EFFORT = {EFF: EFF}
sys.argv = [os.path.join(ROOT, 'serve', 'server.py')] + sys.argv[1:]
runpy.run_path(sys.argv[0], run_name='__main__')
