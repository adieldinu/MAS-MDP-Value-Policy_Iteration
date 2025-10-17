From the graphs:

Gauss-Seidel seems to converge the fastest as it skips over some redundant steps from Value Iteration. It may prove useful when there is the need for a faster convergence.

The same can't be said about Prioritized Sweeping.
	Although i have tried many times it seems it either doesn't converge on some cases or gets blocked on a local plateua, wabbling around a value.

Meanwhile, for Policy Iteration we plot the median out of 5 tries, and it presents a much nicer behaviour:
	On FrozenLake, or FrozenLake8x8 it often does not learn, plateaus imediately at a big cost (The pngs without "-success")
	However, when it finds a good start it is close to Gauss-Seidel.
	
	A much better behaviour is exhibited in simulating TaxiV3.
		Because the environment is bigger and has more actions, the effect of the random start tends to
			be attenuated the further we simulate
		As a result, the loss curve gets away from the Gauss Seidel in the first few iterations, but
			immediately recovers and gets to the optimal solution much faster than the classical Value Iteration.
	
	Sometimes it even surpasses Gauss-Seidel, when it probably lucks into a multitude of good actions from the initial steps.
		Jumping over numerous steps of exploration.
