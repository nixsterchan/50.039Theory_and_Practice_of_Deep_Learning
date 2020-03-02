Nigel Chan Terng Tseng
1002027
ISTD Senior


######## The following are just some little instructions ########

1. The file "DL2_Coding.ipynb" contains the jupyter notebook with code for Task 1 and Task 2.

2. Some dependencies you want to note for the environment (python 3.7) will be:
	- torch
	- numpy
	- matplotlib
	- time

3. Just follow through as I have documented each part in the notebook! If you have cudas enabled, you can uncomment the block that contains the code for using GPU for torch.


######## Observations Made ########

1. After testing out for loops, numpy broadcasting and pytorch CPU broadcasting, pytorch broadcasting performed the best and was about 2 times faster than numpy broadcasting.

2. Testing out with pytorch GPU showed a significant increase in speed. Almost 17 times the speed of pytorch CPU for one case. Observation made here was that the first run of the code for pytorch GPU was around the same or less time than pytorch CPU. However after the memory was cached in GPU, running the block of code once again showed a vast increase in computation speed. This is likely due to the first run having to take time to copy the tensors from CPU to GPU. However, after the tensors were in GPU memory, the second time was way faster.