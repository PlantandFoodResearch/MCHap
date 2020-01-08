import numpy as np
import scipy
from numba import jit

from haplohelper.step import util



def point_beta_probabilities(n_base, a=1, b=1):
    """Return probabilies for selecting a recombination point
    following a beta distribution

    Parameters
    ----------
    n_base : int
        Number of base positions in this genotype.
    a : float
        Alpha parameter for beta distribution.
    b : float
        Beta parameter for beta distribution.

    Returns
    -------
    probs : array_like, int, shape (n_base - 1)
        Probabilities for recombination point.
    
    """
    dist = scipy.stats.beta(a, b)
    points = np.arange(1, n_base ) / (n_base - 1)
    probs = dist.cdf(points)
    probs[1:] = probs[1:] - probs[:-1]
    return probs


@jit(nopython=True)
def recombination_step_n_options(labels):
    """Calculate number of unique haplotype recombination options.
    
    Parameters
    ----------
    labels : array_like, int, shape (n_haps, 2)
        Labels for haplotype sections (excluding duplicated haplotypes).

    Returns
    -------
    n : int
        Number of unique haplotype recombination options

    Notes
    -----
    Duplicate copies of haplotypes must be excluded from the labels array.

    """
    ploidy = len(labels)
    n = 0
    for h_0 in range(ploidy):
        for h_1 in range(h_0 + 1, ploidy):
            if (labels[h_0, 0] == labels[h_1, 0]) or (labels[h_0, 1] == labels[h_1, 1]):
                # this will result in equivilent genotypes
                pass
            else:
                n += 1
    return n


@jit(nopython=True)
def recombination_step_options(genotype, interval=None):
    """Possible recombinations between pairs of unique haplotypes.

    Parameters
    ----------
    genotype : array_like, int, shape (ploidy, n_base)
        Set of haplotypes with base positions encoded as simple integers from 0 to n_nucl.
    interval : array_like, int, size 2
        Lower and upper bounds of the recombined region. May use negative indicies.

    Returns
    -------
    options : array_like, int, shape (n_options, 2)
        Possible recombinations between pairs of unique haplotypes based on the dosage.

    """
    ploidy, n_base = genotype.shape
    
    #labels for the mutating section and remander 
    labels = np.zeros((ploidy, 2), np.int8)
    
    # temprarily use label array to store dosage
    # the dosage is used as a simple way to skip compleately duplicated haplotypes
    util.get_dosage(labels[:,0], genotype)
    # boolean mask for unique haplotypes
    duplicate = labels[:,0] == 0
    
    #labels
    util.label_haplotypes(labels[:,0], genotype, interval)
    # TODO: clean up code
    mask = util._interval_inverse_mask(interval, n_base)
    util.label_haplotypes(labels[:,1], genotype[:, mask])
    
    # remove extra copies of fully duplicate haplotypes 
    labels = labels[~duplicate]
    
    # calculate number of options to create empty array
    n_options = recombination_step_n_options(labels)
    options = np.zeros((n_options, 2), np.int8)
    
    # populate array with options
    opt = 0
    for h_0 in range(ploidy):
        for h_1 in range(h_0 + 1, ploidy):
            if (labels[h_0, 0] == labels[h_1, 0]) or (labels[h_0, 1] == labels[h_1, 1]):
                # this will result in equivilent genotypes
                pass
            else:
                options[opt, 0] = h_0
                options[opt, 1] = h_1
                opt+=1

    return options

@jit(nopython=True)
def log_likelihood_recombination(reads, genotype, h_x, h_y, point):
    """Log likelihood of observed reads given recombination point and pair of 
    haplotypes to recombine at that point.

    Parameters
    ----------
    reads : array_like, float, shape (n_reads, n_base, n_nucl)
        Observed reads encoded as an array of probabilistic matrices.
    genotype : array_like, int, shape (ploidy, n_base)
        Set of haplotypes with base positions encoded as simple integers from 0 to n_nucl.
    h_x : int
        Index of haplotype x in genotype to recombine with haplotype y.
    h_y : int
        Index of haplotype y in genotype to recombine with haplotype x.
    point : int
        Base position at which recombination should occur.

    Returns
    -------
    llk : float
        Log-likelihood of the observed reads given the genotype with recombined haplotypes.

    Notes
    -----
    Recombination point should be a value in range(1, n_base) where n_base is the 
    number of base positions in the genotype.

    """

    ploidy, n_base = genotype.shape
    n_reads = len(reads)
       
    llk = 0.0
    
    for r in range(n_reads):
        
        read_prob = 0
        
        for h in range(ploidy):
            read_hap_prod = 1.0
            
            for j in range(n_base):
                
                if h == h_x and j >= point:
                    i = genotype[h_y, j]
                elif h == h_y and j >= point:
                    i = genotype[h_x, j]
                else:
                    i = genotype[h, j]

                read_hap_prod *= reads[r, j, i]
                
            read_prob += read_hap_prod/ploidy
        
        llk += np.log(read_prob)
                    
    return llk


@jit(nopython=True)
def recombine(genotype, h_x, h_y, point):
    """Recombine a pair of haplotypes within a genotype

    Parameters
    ----------
    genotype : array_like, int, shape (ploidy, n_base)
        Set of haplotypes with base positions encoded as simple integers from 0 to n_nucl.
    h_x : int
        Index of haplotype x in genotype to recombine with haplotype y.
    h_y : int
        Index of haplotype y in genotype to recombine with haplotype x.
    point : int
        Base position at which recombination should occur.
    
    Returns
    -------
    None

    Notes
    -----
    Variables `genotype` is updated in place.

    """
    _, n_base = genotype.shape

    # swap bases from the recombination point
    for j in range(point, n_base):
        
        j_x = genotype[h_x, j]
        j_y = genotype[h_y, j]
        
        genotype[h_x, j] = j_y
        genotype[h_y, j] = j_x


@jit(nopython=True)
def recombination_step(genotype, reads, dosage, llk, point):
    """Recombination Gibbs sampler step between two non-identical haplotypes within a genotype at 
    a given recombination point.

    Parameters
    ----------
    genotype : array_like, int, shape (ploidy, n_base)
        Initial state of haplotypes with base positions encoded as simple integers from 0 to n_nucl.
    reads : array_like, float, shape (n_reads, n_base, n_nucl)
        Observed reads encoded as an array of probabilistic matrices.
    dosage : array_like, int, shape (ploidy)
        Array of initial dosages.        
    llk : float
        Log-likelihood of the initial haplotype state given the observed reads.
    point : int
        Integer index of recombination point which should be a value in range(1, n_base).
    
    Returns
    -------
    llk : float
        New log-likelihood of observed reads given the updated genotype dosage.

    Notes
    -----
    Variables `genotype` and `dosage` are updated in place.

    """
    # recombination options
    # does not include the option of no recombination
    recomb_options = recombination_step_options(dosage)

    # total number of options including no recombination
    n_options = len(recomb_options) + 1

    # log liklihood for each recombination option
    llks = np.empty(n_options)
    
    # iterate through recombination options and calculate log-likelihood
    for opt in range(n_options -1):
        llks[opt] = log_likelihood_recombination(
            reads, 
            genotype, 
            recomb_options[opt, 0],
            recomb_options[opt, 1],
            point
        )

    # final option is to keep the initial genotype (no recombination)
    llks[-1] = llk

    # calculated denominator in log space
    log_denominator = llks[0]
    for opt in range(1, n_options):
        log_denominator = util.sum_log_prob(log_denominator, llks[opt])

    # calculate conditional probabilities
    conditionals = np.empty(n_options)
    for opt in range(n_options):
        conditionals[opt] = np.exp(llks[opt] - log_denominator)

    # ensure conditional probabilities are normalised 
    conditionals /= np.sum(conditionals)
    
    # choose new dosage based on conditional probabilities
    choice = util.random_choice(conditionals)
       
    if choice == (n_options - 1):
        # the state is not changed
        pass
    else:
        # set the state of the recombinated haplotypes
        h_x = recomb_options[choice, 0]
        h_y = recomb_options[choice, 1]
        
        # swap bases from the recombination point
        recombine(genotype, h_x, h_y, point)
                
        # set the new dosage
        util.get_dosage(dosage, genotype)

    # return llk of new state
    return llks[choice]
