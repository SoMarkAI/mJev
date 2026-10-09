"""Candidate accuracy and circular decision consistency."""
import math


def probability_metrics(rows):
    if not rows:
        raise ValueError('Empty metric cohort')
    total_correct = 0
    for row in rows:
        p, target = row['probabilities'], row['target_index']
        if not all(math.isfinite(x) and 0 <= x <= 1 for x in p) or abs(sum(p) - 1) > 1e-5:
            raise ValueError('Invalid candidate probability distribution')
        if not 0 <= target < len(p):
            raise ValueError('Target outside candidates')
        prediction = max(range(len(p)), key=p.__getitem__)
        total_correct += int(prediction == target)
    return {'count': len(rows), 'accuracy': total_correct / len(rows)}


def circular_statistics(groups):
    if not groups:
        raise ValueError('Empty permutation cohort')
    consistent = correct = 0
    for group in groups:
        predictions = [row['semantic_prediction'] for row in group]
        consistent += len(set(predictions)) == 1
        correct += all(row['semantic_prediction'] == row['semantic_target'] for row in group)
    return {'questions': len(groups), 'variants': sum(map(len, groups)),
        'permutation_consistency': consistent / len(groups),
        'circular_accuracy': correct / len(groups),
        'permutation_protocol': 'all cyclic rotations including identity; semantic candidate IDs restored',
        'all_factorial_permutations_tested': False}
