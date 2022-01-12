import numpy as np
from numba import prange, njit, types
from numba.typed import Dict

from ranx.statistical_testing import permute


@njit(cache=True, parallel=True)
def compute_full_avg(c, t):
    rc = c[np.random.randint(c.shape[0])]
    rt = t[np.random.randint(t.shape[0])]

    return abs(rc.mean() - rt.mean()), np.column_stack((rc, rt))


@njit(cache=True, parallel=True)
def compute_avg(c, t):
    r_len = c.shape[1]
    assert r_len == t.shape[1]

    avg_x = np.empty((r_len, 2), dtype=c.dtype)

    for i in prange(r_len):
        avg_x[i, 0] = np.mean(c[:, i])
        avg_x[i, 1] = np.mean(t[:, i])

    return abs(avg_x[:, 0].mean() - avg_x[:, 1].mean()), avg_x


@njit(cache=True, parallel=True)
def compute_full(c, t):
    c_trials, t_trials = c.shape[0], t.shape[0]
    r_len = c.shape[1]
    assert r_len == t.shape[1]

    full_x = np.empty((c_trials * t_trials * r_len, 2), dtype=c.dtype)

    for i in prange(c_trials):
        step = i * c_trials * r_len
        for j in prange(t_trials):
            full_x[step + r_len * j:step + r_len * (j + 1), 0] = c[i]
            full_x[step + r_len * j:step + r_len * (j + 1), 1] = t[j]

    return abs(c.mean() - t.mean()), full_x


@njit(cache=True, parallel=True)
def compute_random_avg(c, t):
    r_len = c.shape[1]
    assert r_len == t.shape[1]

    random_x = np.empty((r_len, 2), dtype=c.dtype)

    for i in prange(r_len):
        random_x[i, 0] = np.random.permutation(c[:, i])[0]
        random_x[i, 1] = np.random.permutation(t[:, i])[0]

    return abs(random_x[:, 0].mean() - random_x[:, 1].mean()), random_x


@njit(cache=True, parallel=True)
def compute_random(c, t):
    return abs(c.mean() - t.mean()), compute_random_avg(c, t)[1]


@njit(cache=True, parallel=True)
def trials_randomization_test(control, treatment, n_permutations=1000, max_p=0.01, random_seed=42, compute='random'):
    ''''
    Performs (approximated) Fisher's Randomization Test.

    Null hypotesis: system A (control) and system B (treatment) are identical (i.e., system A has no effect compared to system B on the mean of a given performance metric)

    For further details, see Smucker et al. A Comparison of Statistical Significance Tests for Information Retrieval Evaluation, CIKM '07.

    '''
    np.random.seed(random_seed)

    control_mean = control.mean()
    treatment_mean = treatment.mean()

    counter_array = np.zeros(n_permutations)

    for i in prange(n_permutations):
        if compute == 'random':
            control_treatment_diff, control_treatment_stack = compute_random(control, treatment)
        elif compute == 'random_avg':
            control_treatment_diff, control_treatment_stack = compute_random_avg(control, treatment)
        elif compute == 'full':
            control_treatment_diff, control_treatment_stack = compute_full(control, treatment)
        elif compute == 'full_avg':
            control_treatment_diff, control_treatment_stack = compute_full_avg(control, treatment)
        elif compute == 'avg':
            control_treatment_diff, control_treatment_stack = compute_avg(control, treatment)
        else:
            raise NotImplementedError

        permuted = permute(control_treatment_stack)

        permuted_diff = abs(permuted[:, 0].mean() - permuted[:, 1].mean())

        if permuted_diff >= control_treatment_diff:
            counter_array[i] = 1.0

    p_value = counter_array.mean()

    return control_mean, treatment_mean, p_value, p_value <= max_p
