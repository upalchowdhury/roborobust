"""Build a portable, offline evidence explorer from existing evaluation artifacts."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from roborobust.replay import load_episode, yaw_from_qpos


def build_showcase(input_dir='results/weekend', output='demo/showcase.html') -> Path:
    source = Path(input_dir)
    raw = (source / 'episode_results.jsonl').read_bytes()
    episodes = [json.loads(line) for line in raw.splitlines() if line.strip()]
    if not episodes:
        raise ValueError('No evaluation episodes found')
    # Only paired seed-zero, pinned scenario-zero traces: no outcome-based selection.
    traces = []
    for row in episodes:
        if row['training_seed'] != 0 or row['scenario_index'] != 0 or row['policy'] not in ('nominal', 'randomized'):
            continue
        path = source / row.get('trajectory_file', '')
        if not path.is_file():
            continue
        arrays, meta = load_episode(path)
        traces.append(dict(policy=row['policy'], condition=row['condition'], scenario=meta['scenario'],
                           reason=row['terminal_reason'], time=arrays['t'].round(4).tolist(),
                           xy=arrays['qpos'][:, :2].round(4).tolist(),
                           yaw=yaw_from_qpos(arrays['qpos']).round(4).tolist(),
                           rays=(arrays['obs'][:, :24] * 3).round(3).tolist()))
    data = dict(episodes=[{k: r[k] for k in ('policy', 'condition', 'training_seed', 'success', 'terminal_reason', 'duration_s')} for r in episodes],
                traces=traces, sha256=hashlib.sha256(raw).hexdigest())
    # Escape HTML-significant characters so artifact contents cannot close the data script.
    payload = json.dumps(data, separators=(',', ':'), allow_nan=False).replace('<', '\\u003c').replace('&', '\\u0026')
    template = Path(__file__).with_name('showcase.html').read_text()
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(template.replace('__EVIDENCE__', payload))
    return out
