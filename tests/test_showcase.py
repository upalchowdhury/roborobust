import json
import re

from roborobust.showcase import build_showcase


def test_showcase_embeds_exact_results_and_pinned_pairs(tmp_path):
    out = build_showcase(output=tmp_path / 'demo.html')
    text = out.read_text()
    data = json.loads(re.search(r'<script id="evidence" type="application/json">(.*?)</script>', text, re.S)[1])
    assert len(data['episodes']) == 3600
    assert len(data['traces']) == 12
    for trace in data['traces']:
        assert len(trace['time']) == len(trace['xy']) == len(trace['rays']) == len(trace['yaw'])
        assert len(trace['rays'][0]) == 24
    stress = [r for r in data['episodes'] if r['condition'] == 'combined_stress']
    for policy, expected in [('nominal', .760), ('randomized', .764)]:
        rows = [r for r in stress if r['policy'] == policy]
        assert sum(r['success'] for r in rows) / len(rows) == expected
    assert '__EVIDENCE__' not in text
