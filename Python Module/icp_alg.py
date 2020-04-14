import numpy as np
import time

# Constants
N = 5                                    # number of random points in the dataset
num_tests = 1                             # number of test iterations
dim = 3                                     # number of dimensions of the points
noise_sigma = .004                           # standard deviation error to be added
translation = .1                            # max translation of the test set
rotation = .1                               # max rotation (radians) of the test set


def best_fit_transform(A, B):
    '''
    Calculates the least-squares best-fit transform that maps corresponding points A to B in m spatial dimensions
    Input:
      A: Nxm numpy array of corresponding points
      B: Nxm numpy array of corresponding points
    Returns:
      T: (m+1)x(m+1) homogeneous transformation matrix that maps A on to B
      R: mxm rotation matrix
      t: mx1 translation vector
    '''

    assert A.shape == B.shape

    # get number of dimensions
    m = A.shape[1]

    # translate points to their centroids
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    AA = A - centroid_A
    BB = B - centroid_B

    # rotation matrix
    H = np.dot(AA.T, BB)
    U, S, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)

    # special reflection case
    if np.linalg.det(R) < 0:
        Vt[m-1, :] *= -1
        R = np.dot(Vt.T, U.T)

    # translation
    t = centroid_B.T - np.dot(R, centroid_A.T)

    # homogeneous transformation
    T = np.identity(m+1)
    T[:m, :m] = R
    T[:m, m] = t

    return T, R, t


def test_best_fit():

    # Generate a random dataset
    #A = np.random.rand(N, dim)
    A = np.array([[0.314147, -0.145258, 1.1335],[0.0846287, -0.175077, 1.03769],[0.316068, -0.365857, 0.834905],
                  [0.216872, -0.104483, 0.853936],[0.225818, -0.208735, 1.08773]])

    '''
    coordinates in space= 0.314147, -0.145258, 1.1335
x img col=718   y img col=266
coordinates in space= 0.0846287, -0.175077, 1.03769
x img col=923   y img col=80
coordinates in space= 0.316068, -0.365857, 0.834905
x img col=837   y img col=298
coordinates in space= 0.216872, -0.104483, 0.853936
x img col=805   y img col=250
coordinates in space= 0.225818, -0.208735, 1.08773
    '''

    total_time = 0
    for i in range(num_tests):

        B = np.array([[0.379184, 0.375408, 0.410453],[0.502678, 0.605722, 0.432820],[0.194563, 0.589923, 0.629036],
                      [0.268782, 0.649020, 0.379206],[0.458242, 0.460640, 0.485377]])

        # Find best fit transform
        start = time.time()
        T, R1, t1 = best_fit_transform(B, A)
        total_time += time.time() - start

        # Make C a homogeneous representation of B
        C = np.ones((N, 4))
        C[:, 0:3] = B

        # Transform C
        C = np.dot(T, C.T).T

        assert np.allclose(C[:, 0:3], A, atol=6*noise_sigma)    # T should transform B (or C) to A
        #assert np.allclose(-t1, t, atol=6*noise_sigma)          # t and t1 should be inverses
        #assert np.allclose(R1.T, R, atol=6*noise_sigma)         # R and R1 should be inverses
    #print(T)
    #print('best fit time: {:.3}'.format(total_time/num_tests))

    return T



#if __name__ == "__main__":
 #   test_best_fit()
