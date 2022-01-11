import numpy as np
from numba import prange

from ranx.statistical_testing import permute


def compute_full(c, t):
    r_c = np.concatenate(np.concatenate([c for _ in c]))
    r_t = np.concatenate(np.concatenate([np.roll(t, i, axis=0) for i, _ in enumerate(t)]))

    return np.column_stack((r_c, r_t))
    

# @njit(cache=True, parallel=True)
def trials_randomization_test(control, treatment, n_permutations=1000, max_p=0.01, random_seed=42):
    ''''
    Performs (approximated) Fisher's Randomization Test.

    Null hypotesis: system A (control) and system B (treatment) are identical (i.e., system A has no effect compared to system B on the mean of a given performance metric)

    For further details, see Smucker et al. A Comparison of Statistical Significance Tests for Information Retrieval Evaluation, CIKM '07.

    '''
    np.random.seed(random_seed)

    control_mean = control.mean()
    treatment_mean = treatment.mean()
    control_treatment_diff = abs(control_mean - treatment_mean)
    control_treatment_stack = compute_full(control, treatment)

    counter_array = np.zeros(n_permutations)

    for i in prange(n_permutations):
        permuted = permute(control_treatment_stack)

        permuted_diff = abs(permuted[:, 0].mean() - permuted[:, 1].mean())

        if permuted_diff >= control_treatment_diff:
            counter_array[i] = 1.0

    p_value = counter_array.mean()

    return control_mean, treatment_mean, p_value, p_value <= max_p
