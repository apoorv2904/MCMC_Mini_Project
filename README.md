# MCMC_Mini_Project
Description

The  core  of  the  code  is  formed  of  the  classes MCMC, Schedule, Experiment,  and MultiExperiments. MCMC models the Markov chain and allows to draw samples with either Metropolis-Hastings or Glauber-dynamics.   It  is  also  used  to  randomly  draw X and W and  to  compute  the  error e(x_est,X).Schedule returns values of  β for a given number N of time steps and a specified cooling strategy. Experiment runs a MCMC experiment with a given set of parameters (n,α,N, cooling schedule) and returns the final error and other statistics.MultiExperiments facilitates running multiple experiments to compare different parameter settings and estimate the mean of the error and its standard deviation. These classes can be found in the files - schedule.py, mcmc.py, experiment.py. The code also contains utils.py which has functions for plotting and saving.


Dependencies:

The code has following dependencies to generate the plots:

Plotly version >= 3.2.0. 
orca 
psutil 

These can be installed by following the instructions over here
https://plot.ly/python/static-image-export/



Generating Results:

The code is divided into following main ipython files to get different results

1. main_plot_annealing_schedules.ipynb - This plots temperature values over time for different cooling strategies.
2. main_annealing_comparison.ipynb - This contains code to compare different cooling schedules and parameters such as beta and number of steps in MCMC.
3. main_metropolis_glauber_comparison.ipynb - This compares Metropolis-Hastings against Glauber-dynamics and also plots error trend over time.
4. main_alpha_critical.ipynb - This generates the plot for critical values of alpha



© 2018 GitHub, Inc.
Terms
Privacy
Security
Status
Help
Contact GitHub
Pricing
API
Training
Blog
About
