import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

# generate homogeneous poisson process 
# with rate lambda0 up to time t
def homogeneous_poisson_times(t, T, lambda0):
    if t > T:
        raise ValueError("t must be less than or equal to T")
    nevents = np.random.poisson(lambda0 * (T - t))
    times = np.random.uniform(t, T, nevents)
    return np.sort(times)

# thinning to generate inhomogeneous poisson process
def thinning(events, ratio):
    return events[np.random.uniform(size=events.shape[0]) < ratio]

def lambdap(lambda0p, alpha, beta, eventtimes, t):
    if eventtimes.size == 0:
        return lambda0p
    
    if not np.isscalar(t):
        t = t.reshape(-1,1)
        eventtimes = eventtimes.reshape(1,-1)
    
    if np.sum(t - eventtimes < 0) > 0:
        raise ValueError("t must be greater than all event times")
    
    if not np.isscalar(t):
        return lambda0p + alpha * beta * np.sum(np.exp(-beta * (t - eventtimes)), axis=1)
    return lambda0p + alpha * beta * np.sum(np.exp(-beta * (t - eventtimes)))
    

# define baseline intensities parameters
lambda0p = 0.3
lambda0n = 1

# define exponential hawkes kernel parameters
alpha = 0.5
beta = 1

# start time and end time
t = 0
T = 400

# initial spread
S = [3]

# initialize list of event times
tau = [0]

# initialize list of spread-opening event times
eventtimesp = np.array([])

# generate homogeneous event times for spread-closing events
eventtimesn = homogeneous_poisson_times(0, T, lambda0n)

while True:
    # calculate lambda^plus_max
    if eventtimesp.size == 0:
        lambdap_max = lambda0p
    else:
        lambdap_max = lambdap(lambda0p, alpha, beta, eventtimesp, eventtimesp[-1])

    # generate homogeneous event times for spread opening events
    eventtimesp_new = homogeneous_poisson_times(t, T, lambdap_max)

    # thin event times
    lambdap_t = lambdap(lambda0p, alpha, beta, eventtimesp, eventtimesp_new)
    eventimesp_new = thinning(eventtimesp_new, lambdap_t / lambdap_max)

    # check if no more events
    if eventimesp_new.size == 0 or eventimesp_new[0] >= T:
        break

    # there is another spread opening event
    else:
        t = eventtimesp_new[0]
        eventtimesp = np.append(eventtimesp, t)
    
    while S[-1] > 1 and eventtimesn.size > 0 and eventtimesn[0] <= t:
        S.append(S[-1] - 1)
        tau.append(eventtimesn[0])
        eventtimesn = np.delete(eventtimesn, 0)

    S.append(S[-1] + 1)
    tau.append(t)

name = "spread_lambda_{}_{}_alpha_{}_beta_{}_T_{}.png".format(lambda0p, lambda0n, alpha, beta, T)
plt.figure()
plt.step(tau, S, label="Spread")
plt.xlabel("Time")
plt.ylabel("Ticks")
plt.legend()
plt.savefig(name, dpi=400)
