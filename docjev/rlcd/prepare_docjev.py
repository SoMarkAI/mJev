"""Connected document/image groups and immutable bilingual question records."""
import collections


def components(documents):
    parent = list(range(len(documents)))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    def union(a, b):
        parent[find(a)] = find(b)
    papers, images = {}, {}
    for i, doc in enumerate(documents):
        paper = doc.get('paper_id') or 'image:' + doc['image_sha256']
        for key, mapping in [(paper, papers), (doc['image_sha256'], images)]:
            if key in mapping:
                union(i, mapping[key])
            else:
                mapping[key] = i
    groups = collections.defaultdict(list)
    for i, doc in enumerate(documents):
        groups[find(i)].append(doc)
    return sorted(groups.values(), key=lambda group: min(d['id'] for d in group))


def paired_rows(doc, media_path, split, source_sha, provenance='codex_generated_bilingual_model_screened_reference'):
    zh, en, ids = doc['questions'], doc['questions_en'], doc['question_ids']
    if not len(zh) == len(en) == len(ids) or not zh:
        raise ValueError('Bilingual question IDs/counts differ')
    pairs = []
    for i, (a, b, qid) in enumerate(zip(zh, en, ids, strict=True)):
        if a['image'] != doc['image'] or b['image'] != doc['image']:
            raise ValueError('Question media differs from document')
        ac, bc = a['question']['criteria'], b['question']['criteria']
        if list(ac) != list(bc) or a['label'] != b['label']:
            raise ValueError('Bilingual label alignment differs')
        if not 2 <= len(ac) <= 128 or a['label'] not in ac:
            raise ValueError('Invalid candidate count/label')
        pair = []
        for language, original in [('zh', a), ('en', b)]:
            pair.append({'id': f'docjev:{qid}:{language}', 'pair_id': qid,
                'dataset': doc['source_dataset'], 'source': doc['source_dataset'],
                'task': 'choice', 'modality': 'image', 'language': language,
                'media_path': media_path, 'media': {'path': media_path},
                'question': original['question']['instructions'],
                'candidates': original['question']['criteria'], 'label': original['label'],
                'split': split, 'image_sha256': doc['image_sha256'],
                'original_image': doc['image'], 'paper_id': doc.get('paper_id'),
                'source_question_index': i, 'source_rows': doc['source_rows'],
                'source_document_sha256': source_sha,
                'reference_provenance': provenance})
        pairs.append(pair)
    return pairs
