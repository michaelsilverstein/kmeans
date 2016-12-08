def fuzzifier(point,cluster_coordinates):
    #Produces vector of probabilities of each point being associated with a given cluster
    u = cluster_coordinates
    all_probs = [[gauss_prob(point,k)] for k in u]
    fuzz_vector = [gauss_prob(point,k)/np.sum(all_probs) for k in u]
    return fuzz_vector
def k_means(data_coordinates,labels,cluster_coordinates):
    #Performs k-means algorithm
    """
    Estimation step:
    Assign each data point to the nearest cluster
    |labels = P(labels|data,parameters)
    |d = (x_i-u_k)^2, for data point,x_i, and cluster, u_k
    |np.linalg.norm: Computes the norm(x_i,u_k)
    |pow(#,2): raises # to second power
    |np.argmin(Z) for k in u: return the position, k in u, that minimizes Z
    |for i in p: do all of these for each coordinate
    ***********************************************
    Maximization step:
    Move cluster to the centroid of the points with its respective label
    |np.sum(p[a,x]), where a is position of the data point in the matrix and
    x is the coordinate (x and then y)
    |len(l for l in labels if l==k), computes how many points are in a given label assignment
    |for a in range(len(p)) if int(label[a]==k): filters through points for each cluster
    |for k in range(len(u)): compute this for each cluster
    ******************************************************************
    Stopping Condition:
    |Primary: if the labels do not change from iteration n-1 to n, stop
    |Secondary: If the loop exceeds 100 iterations
    """
    p = data_coordinates
    u = cluster_coordinates
    old_labels = np.array([-1] * len(p))  # initialize
    for _ in range(100):
        #E step
        labels = np.array([np.argmin([pow(np.linalg.norm(i-k),2)for k in u])for i in p])
        #M step
        u = np.array([p[labels==k].mean(axis=0) for k in range(len(u))])
        # if (labels[x] == old_labels[x] for x in range(len(labels))):  # Stopping criteria
        #     print 'Reached convergence...'
        #     return labels, np.array(u)
        # else:
        #     old_labels = labels
    return labels,u
def fuzzy_k_means(data_coordinates,labels,cluster_coordinates):
    #Performs fuzzy k means algorithm
    """
    Estimation Step:
    Uses the fuzzifier function to create a vector of the probability of a given point
    belonging to each cluster
    Ex. 2 data points, 2 clusters: labels = [[0.89, 0.10],[0.4,0.7]]
    """
    p = data_coordinates
    u = cluster_coordinates
    tolerance = 0.01
    move_distances = [1000]*len(u)
    for _ in range(100):
        #E step
        labels = np.array([fuzzifier(point,u) for point in p])
        # print labels

        #M Step
        u_new = [[np.sum(p[:,x]*labels[:,k]/np.sum(labels[:,k])) for x in range(2)]for k in range(len(u))]

        d = [distance(u[k], u_new[k]) for k in range(len(u))]
        if min(d) <= float(0):
            final_labels = np.array([np.argmax(label) for label in labels])
            return final_labels, np.array(u_new)
        for k in range(len(d)):
            diff = int(abs(move_distances[k] - d[k]) / d[k])
            if diff < tolerance:
                final_labels = np.array([np.argmax(label) for label in labels])
                return final_labels, np.array(u_new)
            move_distances[k] = d[k]
        u = u_new
