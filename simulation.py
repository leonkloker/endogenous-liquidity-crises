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
    return sorted(times)

# calculate lambda(t) for exponential hawkes process
def lambdap(lambda_0, alpha, beta, past_events, t):
    past_events = np.array(past_events)
    if np.sum(t < past_events) > 0:
        raise ValueError("t must be greater than all past event times")
    
    return lambda_0 + alpha * beta * np.sum(np.exp(-beta * (t - past_events)))
    

# define baseline intensities parameters
lambda_0_p = 0.3
lambda_0_n = 1

# define exponential hawkes kernel parameters
alpha = 1.001
beta = 1

# start time and end time
t = 0
T = 100

# initial spread
S = [20]

# initialize list of event times
tau = [0]

# initialize list of spread-opening event times
times_p = []

# generate homogeneous event times for spread-closing events
times_n = homogeneous_poisson_times(0, T, lambda_0_n)

# maximal lambda_p
lambda_p_max = lambda_0_p

# generate spread opening events independent of spread closing events
while t < T:
    
    # generate homogeneous event times for spread opening events
    times_p_new = homogeneous_poisson_times(t, T, lambda_p_max)

    newmax = False

    while not newmax:

        # check if there are any events
        if len(times_p_new) == 0:
            t = T
            break

        # take the first event time t
        t = times_p_new.pop(0)

        # calculate lambda(t)
        lambda_t = lambdap(lambda_0_p, alpha, beta, times_p, t)

        # check if event occurs
        if np.random.uniform(0, 1) < lambda_t / lambda_p_max:
            times_p.append(t)

            # check if lambda(t) is new max due to this event
            if lambda_t + alpha*beta > lambda_p_max:
                lambda_p_max = lambda_t + alpha*beta
                newmax = True

# merge spread opening and closing events
while len(times_n) > 0:

    # add opening events until next closing event
    while len(times_p) > 0 and times_p[0] < times_n[0]:
        tau.append(times_p.pop(0))
        S.append(S[-1] + 1)

    # add closing event only if spread is open
    if S[-1] > 1:
        tau.append(times_n.pop(0))
        S.append(S[-1] - 1)

    # remove the closing event if spread is closed
    else:
        times_n.pop(0)

# add remaining opening events
while len(times_p) > 0:
    tau.append(times_p.pop(0))
    S.append(S[-1] + 1)

name = "spread_lambda_{}_{}_alpha_{}_beta_{}_T_{}.png".format(lambda_0_p, lambda_0_n, alpha, beta, T)
plt.figure()
plt.step(tau, S, label="Spread")
plt.xlabel("Time")
plt.ylabel("Ticks")
plt.legend()
plt.savefig(name, dpi=400)
