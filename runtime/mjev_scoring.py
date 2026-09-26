"""Shared, parameter-free candidate normalization and deterministic decision."""
import math
from mjev.decision import choose

def score_answer(question_id, scores):
    if len(scores)<2 or any(not math.isfinite(v) for v in scores.values()):
        raise ValueError('Expected at least two finite raw candidate logits')
    peak=max(scores.values());weights=[math.exp(v-peak) for v in scores.values()];total=sum(weights)
    candidates=[{'label':label,'raw_logit':float(value),'probability':weight/total}
                for (label,value),weight in zip(scores.items(),weights)]
    decision=choose(candidates)
    return {'id':question_id,'answer':decision['label'],'scores':scores,
            'probabilities':{c['label']:c['probability'] for c in candidates},
            'candidates':candidates,'decision':decision,
            'probability_sum':sum(c['probability'] for c in candidates)}
