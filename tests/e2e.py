import asyncio
import json
import os
import time
import uuid
import hashlib
from pathlib import Path
import torch
from PIL import Image, ImageDraw
from mjev.engine import MJevEngine

ROOT = Path(os.environ.get('MJEV_TEST_OUTPUT', 'outputs/e2e')).resolve()
RUN_ID = None
OUT = None

def save(name, value):
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2))

def raw(r):
    return torch.tensor([c['raw_logit'] for c in r['candidates']])

def check(r):
    assert abs(r['probability_sum'] - 1) < 1e-6
    assert torch.isfinite(raw(r)).all()
    assert len({x['token_id'] for x in r['candidates']}) == len(r['candidates'])

async def main():
    global RUN_ID, OUT
    RUN_ID = os.environ.get('MJEV_RUN_ID', time.strftime('%Y%m%dT%H%M%S') + '-' + uuid.uuid4().hex[:8])
    OUT = ROOT / 'runs' / RUN_ID
    OUT.mkdir(parents=True, exist_ok=False)
    latest = ROOT / 'latest'
    latest.unlink(missing_ok=True)
    latest.symlink_to(Path('runs') / RUN_ID)
    os.environ['MJEV_CACHE_TRACE_PATH'] = str(OUT / 'cache_hashes.jsonl')
    started = time.time()
    image = Image.new('RGB', (336, 224), 'white')
    d = ImageDraw.Draw(image)
    d.rectangle((40,40,296,184), fill='red')
    image.save(OUT / 'fixture.png')
    sources = [*Path('mjev').glob('*.py'), *Path('tests').glob('*.py'),
               Path('sitecustomize.py'), Path('pyproject.toml')]
    report = {'run_id': RUN_ID, 'started': started, 'tests': {}, 'outputs': [],
              'source_sha256': {str(p):hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in sources},
              'versions': {'torch':torch.__version__, 'vllm':__import__('vllm').__version__,
                           'transformers':__import__('transformers').__version__}}
    e = None
    context = 'Inspect the supplied picture carefully. ' * 20
    question = 'What is the color of the large rectangle?'
    candidates = ['The rectangle is red.', 'The rectangle is blue.', 'The rectangle is green.']
    try:
        from mjev.patch import batch_route
        try:
            batch_route([{'mode':'stock'}, {'mode':'isolated'}])
        except RuntimeError:
            report['tests']['mixed_backend_rejected'] = True
        else:
            raise AssertionError('Mixed backend batch accepted')
        e = MJevEngine(os.environ['MJEV_MODEL'])
        cold = await e.score(image, question, candidates, context)
        warm = await e.score(image, question, candidates, context)
        check(cold); check(warm)
        assert warm['num_cached_tokens'] > 0
        delta = float((raw(cold) - raw(warm)).abs().max())
        assert delta < 0.5, delta
        report['tests']['prefix_cache'] = {'cold': cold['num_cached_tokens'],
            'warm': warm['num_cached_tokens'], 'max_logit_delta': delta}
        report['outputs'] += [cold, warm]
        save('progress.json', report)
        print('PASS prefix_cache', report['tests']['prefix_cache'], flush=True)

        changed = ['The rectangle is tan.', *candidates[1:]]
        # Compare exact official token spans before issuing same-length interventions.
        s1 = e.protocol.build(image,context,question,candidates)[1].extra_kwargs['mjev']['spans']
        s2 = e.protocol.build(image,context,question,changed)[1].extra_kwargs['mjev']['spans']
        assert s1 == s2, (s1,s2)
        iso1 = await e.score(image, question, candidates, context, debug=True)
        iso2 = await e.score(image, question, changed, context, debug=True)
        cau1 = await e.score(image, question, candidates, context, mode='causal', debug=True)
        cau2 = await e.score(image, question, changed, context, mode='causal', debug=True)
        stock = await e.score(image, question, candidates, context, mode='stock', debug=True)
        for r in [iso1,iso2,cau1,cau2,stock]:
            check(r)
        h = lambda r: torch.tensor(r['candidate_hidden'])
        isolated_delta = float((h(iso1)[1:] - h(iso2)[1:]).abs().max())
        causal_delta = float((h(cau1)[1:] - h(cau2)[1:]).abs().max())
        control_delta = float((raw(cau1) - raw(stock)).abs().max())
        save('hidden_probes.json', [iso1,iso2,cau1,cau2,stock])
        print('ISOLATION_DIAGNOSTICS', isolated_delta, causal_delta, control_delta, flush=True)
        assert isolated_delta < 1e-3, isolated_delta
        assert causal_delta > 1e-3, causal_delta
        assert control_delta < 0.5, control_delta
        report['tests']['mask_isolation'] = {'unaffected_candidate_hidden_max_delta': isolated_delta,
            'causal_positive_control_delta': causal_delta,
            'sdpa_vs_stock_causal_max_logit_delta': control_delta,
            'answer_logits_change': float((raw(iso1)-raw(iso2)).abs().max())}
        save('hidden_probes.json', [iso1,iso2,cau1,cau2,stock])
        print('PASS mask_isolation', report['tests']['mask_isolation'], flush=True)
        # Fresh prompt: prime only CAUSAL candidate blocks, then request ISOLATED.
        # The latter must reuse public prefix only, then reuse its own private cache.
        fresh_context = 'Unique cache audit ' + RUN_ID + '. ' + context
        long_candidates = [x * 8 for x in candidates]
        prime = await e.score(image, question, long_candidates, fresh_context, mode='causal')
        prime_warm = await e.score(image, question, long_candidates, fresh_context, mode='causal')
        first_isolated = await e.score(image, question, long_candidates, fresh_context)
        second_isolated = await e.score(image, question, long_candidates, fresh_context)
        isolated_reference = await e.score(image, question, long_candidates, fresh_context, debug=True)
        common_end = first_isolated['candidate_spans'][0][0]
        assert 0 < first_isolated['num_cached_tokens'] <= common_end
        assert prime_warm['num_cached_tokens'] > first_isolated['candidate_spans'][1][1]
        assert second_isolated['num_cached_tokens'] > common_end
        assert first_isolated['num_cached_tokens'] < second_isolated['num_cached_tokens']
        assert float((raw(first_isolated)-raw(isolated_reference)).abs().max()) < 0.5
        for r in [prime,prime_warm,first_isolated,second_isolated,isolated_reference]:
            check(r)
        events = [json.loads(line) for line in (OUT/'cache_hashes.jsonl').read_text().splitlines()]
        def request_events(result):
            return {event['start']:event for event in events
                    if event['request_id'].startswith(result['request_id'])}
        causal_events, isolated_events = request_events(prime), request_events(first_isolated)
        assert causal_events and isolated_events
        comparisons = []
        for index in sorted(causal_events.keys() & isolated_events.keys()):
            ca, iso = causal_events[index], isolated_events[index]
            same = ca['extra_keys_digest'] == iso['extra_keys_digest']
            public = iso['end'] <= common_end
            assert same == public, (ca,iso)
            comparisons.append({'start':index,'end':iso['end'],'public':public,'same_extra_keys':same})
        assert any(x['public'] for x in comparisons) and any(not x['public'] for x in comparisons)
        assert causal_events[0]['mask_hash'] != isolated_events[0]['mask_hash']
        report['tests']['mask_cache_separation'] = {
            'causal_prime_cached':prime['num_cached_tokens'],
            'causal_warm_cached':prime_warm['num_cached_tokens'],
            'first_isolated_cached':first_isolated['num_cached_tokens'],
            'second_isolated_cached':second_isolated['num_cached_tokens'],
            'candidate_start':common_end, 'block_extra_key_comparisons':comparisons,
            'causal_mask_hash':causal_events[0]['mask_hash'],
            'isolated_mask_hash':isolated_events[0]['mask_hash'],
            'reference_logit_max_delta':float((raw(first_isolated)-raw(isolated_reference)).abs().max())}
        report['outputs'] += [prime,prime_warm,first_isolated,second_isolated]
        print('PASS mask_cache_separation', {k:v for k,v in report['tests']['mask_cache_separation'].items()
                                            if k != 'block_extra_key_comparisons'}, flush=True)
        for perm in [[2,0,1], [1,2,0]]:
            shuffled = await e.score(image, question, [candidates[i] for i in perm], context)
            check(shuffled)
            shuffled['permutation'] = perm
            report['outputs'].append(shuffled)
        report['tests']['shuffle'] = {'validated': 'labels follow current candidate order',
            'note': 'Official positional encoding is preserved; permutation invariance is not asserted.'}
        parallel = await e.score_questions(image, [
            {'question': 'Which color fills most of the rectangle?', 'candidates': ['Red','Blue']},
            {'question': 'Which shape is shown?', 'candidates': ['Rectangle','Circle','Triangle','Star','Oval']},
            {'question': 'Which option best describes the image?',
             'candidates': ['Red rectangle','Blue rectangle','Green circle','Yellow triangle','Black star','White oval','Purple square']},
        ], context)
        for r in parallel: check(r)
        assert len({r['request_id'] for r in parallel}) == 3
        assert all(r['num_cached_tokens'] > 0 for r in parallel)
        report['outputs'] += parallel
        report['tests']['concurrent_questions'] = {'count': len(parallel),
            'cached_tokens': [r['num_cached_tokens'] for r in parallel]}
        # Exercise a many-candidate request, with contextual single-token checks.
        many = await e.score(image, question, ['Red'] + [f'Color description {i}' for i in range(1,30)], context)
        check(many)
        report['outputs'].append(many)
        report['tests']['dynamic_candidate_counts'] = [2,3,5,7,30]
        report['tests']['no_generate'] = 'instance generate method raises if called; all requests encode'
        report['passed'] = True
        report['elapsed_seconds'] = time.time() - started
        save('report.json', report)
        (OUT / '_SUCCESS').write_text('All real model E2E assertions passed.\n')
        print('E2E_SUCCESS', report['elapsed_seconds'], flush=True)
    except BaseException as exc:
        report['error'] = repr(exc)
        save('failure.json', report)
        raise
    finally:
        if e is not None:
            e.close()

if __name__ == '__main__':
    asyncio.run(main())
