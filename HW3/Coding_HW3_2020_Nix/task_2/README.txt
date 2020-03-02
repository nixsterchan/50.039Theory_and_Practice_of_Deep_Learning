Nigel Chan Terng Tseng
1002027
ISTD Senior

######## The following are just some little instructions ########

There are two alternatives:
1. If you prefer Jupyter notebook:
	a. Run jupyter notebook on your terminal where the directory is the root directory of 	the “Coding_HW3_2020_Nix” folder

	b. Enter the folder "task_1" and open up "DL3_Coding_Task2.ipynb".

	c. Some dependencies you want to note for the environment (python 3.7) will be the 		same ones that were included within the "pytorch_logreg_gdesc_studentversion.py" code.

2. If you prefer just running the python files:
	a. Change directory in your terminal to the root of the “Coding_HW1_2020_Nix” folder.

	b. Change directory into the "task_2" folder.

	c. Run “python DL3_Coding_Task2.py” to run the training. The results will be seen in 		the terminal.



######### Question ###########
When you train a deep neural net, then you
get after every epoch one model (actually after every minibatch). Why
you should not select the best model over all epochs on the test dataset?

Answer:
It is possible that the best model selected could have an overfit as compared to the other models from other epochs. 