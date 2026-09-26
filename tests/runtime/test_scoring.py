import math
import pytest
from mjev_scoring import score_answer

def test_stability_and_shift_invariance():
    a=score_answer('q',{'A':10000.,'B':9999.,'C':9997.})
    b=score_answer('q',{'A':0.,'B':-1.,'C':-3.})
    assert a['probabilities']==pytest.approx(b['probabilities'])
    assert a['probability_sum']==pytest.approx(1)
    assert a['scores']['A']==10000.

def test_ties_follow_input_order():
    r=score_answer('q',{'A':2.,'B':2.})
    assert r['answer']=='A' and r['decision']['tied_labels']==['A','B']

@pytest.mark.parametrize('bad',[float('nan'),float('inf'),-float('inf')])
def test_nonfinite_rejected(bad):
    with pytest.raises(ValueError):score_answer('q',{'A':bad,'B':0.})
