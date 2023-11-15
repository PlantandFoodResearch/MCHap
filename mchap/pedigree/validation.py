import numpy as np
from numba import njit

from mchap.calling.utils import allelic_dosage
from .prior import parental_copies, initial_dosage, increment_dosage


@njit(cache=True)
def duo_valid(progeny, parent, tau, lambda_):
    dosage = allelic_dosage(progeny)
    dosage_p = parental_copies(parent, progeny)
    constraint_p = np.minimum(dosage, dosage_p)

    # handle lambda parameter (diploid gametes only)
    if lambda_ > 0.0:
        if tau != 2:
            raise ValueError(
                "Non-zero lambda is only supported for a gametic ploidy (tau) of 2"
            )
        # adjust constraint for double reduction
        for i in range(len(dosage)):
            if (dosage[i] >= 2) and (constraint_p[i] == 1):
                constraint_p[i] = 2
    return constraint_p.sum() >= tau


@njit(cache=True)
def trio_valid(
    progeny,
    parent_p,
    parent_q,
    tau_p,
    tau_q,
    lambda_p,
    lambda_q,
):
    dosage = allelic_dosage(progeny)
    dosage_p = parental_copies(parent_p, progeny)
    dosage_q = parental_copies(parent_q, progeny)

    constraint_p = np.minimum(dosage, dosage_p)
    constraint_q = np.minimum(dosage, dosage_q)

    # handle lambda parameter (diploid gametes only)
    if lambda_p > 0.0:
        if tau_p != 2:
            raise ValueError(
                "Non-zero lambda is only supported for a gametic ploidy (tau) of 2"
            )
        # adjust constraint for double reduction
        for i in range(len(dosage)):
            if (dosage[i] >= 2) and (constraint_p[i] == 1):
                constraint_p[i] = 2
    if lambda_q > 0.0:
        if tau_q != 2:
            raise ValueError(
                "Non-zero lambda is only supported for a gametic ploidy (tau) of 2"
            )
        # adjust constraint for double reduction
        for i in range(len(dosage)):
            if (dosage[i] >= 2) and (constraint_q[i] == 1):
                constraint_q[i] = 2

    if (constraint_p.sum() < tau_p) or (constraint_q.sum() < tau_q):
        return False
    gamete_p = initial_dosage(tau_p, constraint_p)
    gamete_q = dosage - gamete_p
    while True:
        match = True
        for i in range(len(dosage)):
            if gamete_q[i] > constraint_q[i]:
                match = False
                break
            if gamete_p[i] + gamete_q[i] != dosage[i]:
                match = False
                break
        if match:
            return True
        try:
            increment_dosage(gamete_p, constraint_p)
        except:  # noqa: E722
            break
        else:
            for i in range(len(gamete_q)):
                gamete_q[i] = dosage[i] - gamete_p[i]
    return False
