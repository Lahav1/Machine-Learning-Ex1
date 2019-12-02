import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from scipy.misc import imread

def init_centroids(X, K):
    """
    Initializes K centroids that are to be used in K-Means on the dataset X.

    Parameters
    ----------
    X : ndarray, shape (n_samples, n_features)
        Samples, where n_samples is the number of samples and n_features is the number of features.
    K : int
        The number of centroids.

    Returns
    -------
    centroids : ndarray, shape (K, n_features)
    """
    if K == 2:
        return np.asarray([[0.        , 0.        , 0.        ],
                            [0.07843137, 0.06666667, 0.09411765]])
    elif K == 4:
        return np.asarray([[0.72156863, 0.64313725, 0.54901961],
                            [0.49019608, 0.41960784, 0.33333333],
                            [0.02745098, 0.        , 0.        ],
                            [0.17254902, 0.16862745, 0.18823529]])
    elif K == 8:
        return np.asarray([[0.01568627, 0.01176471, 0.03529412],
                            [0.14509804, 0.12156863, 0.12941176],
                            [0.4745098 , 0.40784314, 0.32941176],
                            [0.00784314, 0.00392157, 0.02745098],
                            [0.50588235, 0.43529412, 0.34117647],
                            [0.09411765, 0.09019608, 0.11372549],
                            [0.54509804, 0.45882353, 0.36470588],
                            [0.44705882, 0.37647059, 0.29019608]])
    elif K == 16:
        return np.asarray([[0.61568627, 0.56078431, 0.45882353],
                            [0.4745098 , 0.38039216, 0.33333333],
                            [0.65882353, 0.57647059, 0.49411765],
                            [0.08235294, 0.07843137, 0.10196078],
                            [0.06666667, 0.03529412, 0.02352941],
                            [0.08235294, 0.07843137, 0.09803922],
                            [0.0745098 , 0.07058824, 0.09411765],
                            [0.01960784, 0.01960784, 0.02745098],
                            [0.00784314, 0.00784314, 0.01568627],
                            [0.8627451 , 0.78039216, 0.69803922],
                            [0.60784314, 0.52156863, 0.42745098],
                            [0.01960784, 0.01176471, 0.02352941],
                            [0.78431373, 0.69803922, 0.60392157],
                            [0.30196078, 0.21568627, 0.1254902 ],
                            [0.30588235, 0.2627451 , 0.24705882],
                            [0.65490196, 0.61176471, 0.50196078]])
    else:
        print('This value of K is not supported.')
        return None


def print_cent(cent):
    """
    Prints the current centroids.

    Parameters
    ----------
    cent: Centroid array.

    Returns
    -------
    String with the centroid's info in a printing format.
    """
    if type(cent) == list:
        cent = np.asarray(cent)
    if len(cent.shape) == 1:
        return ' '.join(str(np.floor(100*cent)/100).split()).replace('[ ', '[').replace('\n', ' ').replace(' ]',']').replace(' ', ', ')
    else:
        return ' '.join(str(np.floor(100*cent)/100).split()).replace('[ ', '[').replace('\n', ' ').replace(' ]',']').replace(' ', ', ')[1:-1]


def calculate_distance(point1, point2):
    """
    Calculates the euclidean "distance" between two points.

    Parameters
    ----------
    point1, point2.

    Returns
    -------
    the calculated distance between the points.
    """
    r0 = point1[0]
    g0 = point1[1]
    b0 = point1[2]
    r1 = point2[0]
    g1 = point2[1]
    b1 = point2[2]
    # calculate the euclidean distance between the points.
    dist = np.math.sqrt(np.power(r1 - r0, 2) + np.power(g1 - g0, 2) + np.power(b1 - b0, 2))
    return dist


def find_closest_centroid(X, K, centroids):
    """
    First step of K-means algorithm. Orders the pixels in k cluster by the centroids' values.

    Parameters
    ----------
    X: list of pixels.
    K: value of k.
    centroids: list of centroids.

    Returns
    -------
    Prints the values of the centroids after each of the 10 iterations.
    """
    clusters = []
    # iterate 10 times.
    for i in range(1, 11):
        # create a list of k clusters.
        clusters = []
        for j in range(K):
            cluster = []
            clusters.append(cluster)
        # iterate all the pixels.
        for pixel in X:
            # for each point, create a distance list from all centroids.
            distancelist = []
            for cent in centroids:
                distancelist.append(calculate_distance(cent, pixel))
            # find the index of the minimum distance.
            idx = distancelist.index(min(distancelist))
            # add the pixel to the relative cluster.
            clusters[idx].append(pixel)
        # update each centroid to be the average value of its' cluster.
        centroids = update_centroids(clusters)
        # print the current centroid values.
        print("iter %d:" %i, print_cent(centroids))
    return [centroids, clusters]

def update_centroids(clusters):
    """
    Second step of the k-means algorithm. Updates each centroid to be the average of its' cluster.

    Parameters
    ----------
    clusters: list of all clusters.

    Returns
    -------
    Updated list of clusters.
    """
    # create an empty new centroid list.
    centroidlist = []
    # iterate the cluster list.
    for cluster in clusters:
        # for each cluster calculate the average R, G, B values.
        sumR = 0
        sumG = 0
        sumB = 0
        for pixel in cluster:
            sumR += pixel[0]
            sumG += pixel[1]
            sumB += pixel[2]
        avgR = sumR / len(cluster)
        avgG = sumG / len(cluster)
        avgB = sumB / len(cluster)
        # create a new centroid for the current cluster and add it to the new list.
        cent = [avgR, avgG, avgB]
        centroidlist.append(cent)
    return np.asarray(centroidlist)

def compressPicture(A, clustersAndCentroids):
    """
    Replaces all the pixels in each cluster with the centroid and shows the new compressed picture.

    Parameters
    ----------
    A: list of pixels of the original picture.
    clustersAndCentroids: pair of cluster list and centroid list.

    Returns
    -------
    Shows the new compressed picture.
    """
    centroids = clustersAndCentroids[0]
    clusters = clustersAndCentroids[1]
    for a in A:
        for pixel in a:
            for cluster in clusters:
                for p in cluster:
                    if p[0] == pixel[0] and p[1] == pixel[1] and p[2] == pixel[2]:
                        pixel[0] = centroids[clusters.index(cluster)][0]
                        pixel[1] = centroids[clusters.index(cluster)][1]
                        pixel[2] = centroids[clusters.index(cluster)][2]
    plt.imshow(A)
    plt.grid(False)
    plt.show()

def kmeans(A, X):
    """
    For each k, print the k value, initialize the centroids, print the initial values and start the algorithm.
    """
    print("k=2:")
    centroids = init_centroids(X, 2)
    print("iter 0:", print_cent(centroids))
    clustersAndCentroids = find_closest_centroid(X, 2, centroids)
    # compressPicture(A, clustersAndCentroids)

    rl = reload()
    A = rl[0]
    X = rl[1]
    print("k=4:")
    centroids = init_centroids(X, 4)
    print("iter 0:", print_cent(centroids))
    clustersAndCentroids = find_closest_centroid(X, 4, centroids)
    # compressPicture(A, clustersAndCentroids)

    rl = reload()
    A = rl[0]
    X = rl[1]
    print("k=8:")
    centroids = init_centroids(X, 8)
    print("iter 0:", print_cent(centroids))
    clustersAndCentroids = find_closest_centroid(X, 8, centroids)
    # compressPicture(A, clustersAndCentroids)

    rl = reload()
    A = rl[0]
    X = rl[1]
    print("k=16:")
    centroids = init_centroids(X, 16)
    print("iter 0:", print_cent(centroids))
    clustersAndCentroids = find_closest_centroid(X, 16, centroids)
    # compressPicture(A, clustersAndCentroids)

def reload():
    """
    Reloads the picture for the next operation of the k-means algorithm.
    """
    path = 'dog.jpeg'
    A = imread(path)
    A = A.astype(float) / 255.
    img_size = A.shape
    X = A.reshape(img_size[0] * img_size[1], img_size[2])
    return [A, X]


# data preperation (loading, normalizing, reshaping)
path = 'dog.jpeg'
A = imread(path)
A = A.astype(float) / 255.
img_size = A.shape
X = A.reshape(img_size[0] * img_size[1], img_size[2])
# run kmeans algorithm.
kmeans(A, X)



