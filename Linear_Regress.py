import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# This file implments both a linear regression and a PLA on a set of randomly
# generated points in the range [-1, 1] and a randomly generated target function.
# The file aims to show how good of a job PLA and linear regression do. The
# program calculates both in sample error and out of sample error averaged over
# many iterations.

def gen_points(length):
    '''
        Generates and returns an array(of length "length") of vectors
        Parameters:
        length (int) : how many random vectors to generate

        Return:
        np array of vectors : the array of random vectors

    '''
    xn = np.random.uniform(-1, 1, (length, 3))
    for x in xn:
        # We need the 'dummy' value of one for every vector
        x[0] = 1
    return xn

def check_pts(in_set, w):
    '''
        Checks the points in "in_set" agaisnt hypothesis "w".

        Parameters:
        in_set: dictionary where the key is a point in the input space that
        maps to an int that represents the point's target function value

        w : weight vector that represents a hypothesis

        Returns:
        array of point vectors : an array of vectors(points) that did not agree
        with the hypothesis

    '''
    pts = in_set.keys()
    bad_pts = []
    for pt in pts:
        if not check_hypoth(pt, in_set.get(pt), w):
            bad_pts.append(pt)
    return bad_pts

def gen_target():
    '''
        Generates a random target function based on two random points.

        Returns:
        tuple : the first two indices are the randomly generated points and the
        third index is the slope of the generated line.

    '''
    p1 = np.random.random(2)
    p2 = np.random.random(2)
    # Slope between the two points
    return p1, p2, (p1[1] - p2[1]) / (p1[0] - p2[0])

# The target function
def f(x, p1, p2, m):
    '''
        Evaluates a vector in the input space on a target function.

        Parameters:
        x (nparray) : the vector in the input space
        p1 (nparray) : a randomly generated point that was used to generate a
        random target function.
        p2 (nparray) : a randomly generated point that was used to generate a
        random target function.
        m (double) : the slope of the line that represents the target function.

        Returns:
        (int) : The value of the target function on the point x.

    '''
    # if the point x is above the line then the output is 1
    if x[1] >= (m * (x[0]-p1[0]) + p1[1]):
        return 1
    else:
        # If below then output = -1
        return -1

def check_hypoth(x, y, w):
    '''
        Checks a single point agaisnt a hypothesis.

        Parameters:
        x (nparray) : vector in the input space
        y (int) : The value of the target function evauluated at "x"
        w (nparray) : array of weights that represents a hypothesis

        Returns:
        (boolean) : True if the given hypothesis matches the target function
        False if otherwise.
    '''
    sum = 0
    # dot product
    for i in range(len(w)):
        sum += w[i] * x[i]
    if sum >= 0:
        return y > 0
    else:
        return y < 0


iters = 1000 # How many iterations to add over

N = 100 # Number of samples in the input set
M = 1000 # Number of samples out of the set
E_in_sum = 0
E_out_sum = 0
pla_iters_sum = 0
for _ in range(iters):

    # generates target function
    p1, p2, m = gen_target()
    xn = gen_points(N)


    yn = [] # THe outputs of the target function of the input set
    in_set = {}
    for x in xn:
        yn.append(f(x, p1, p2, m))
        x_vect = (x[0], x[1], x[2])
        in_set[x_vect] = f(x, p1, p2, m)
    # Create matrices and perform the linear Regression
    X = np.matrix(xn)
    Y = np.matrix(yn).getT()
    w = np.matmul(np.linalg.pinv(X), Y)

    # Calculate in sample error
    bad_points = []
    for i in range(len(xn)):
        if not check_hypoth(xn[i], yn[i], w):
            bad_points.append(xn[i])
    E_in_sum += len(bad_points) / N


    # Calculate an estimate for the out of sample error
    out_samp = gen_points(M)
    bad_points = []
    out_samp_ys = [] # THe outputs of the target function of the input set
    for x in out_samp:
        out_samp_ys.append(f(x, p1, p2, m))

    for i in range(len(out_samp)):
        if not check_hypoth(out_samp[i], out_samp_ys[i], w):
            bad_points.append(out_samp[i])
    E_out_sum += len(bad_points) / M

    # Do PLA starting with the hypothesis(w) generated from Linear Regression
    bad_pts = check_pts(in_set, w)
    # We assume that tge input set is Linearly Separable otherwise this would
    # loop to infinity.
    while len(bad_pts) > 0:
        pla_iters_sum += 1
        j = np.random.randint(len(bad_pts))
        for i in range(len(w)):
            w[i] += bad_pts[j][i] * in_set.get(bad_pts[j])

        bad_pts = check_pts(in_set, w)




print("Avg E_in: {}".format(E_in_sum / iters))
print("Avg E_out: {}".format(E_out_sum / iters))
print("Avg PLA_iters: {}".format(pla_iters_sum / iters))
