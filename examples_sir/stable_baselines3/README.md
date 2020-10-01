# Example scripts

The plots and models here are the same as used in "One-Shot Epidemic Control with Soft Actor Critic". 

Run the following to create the plots produced in the paper:
`bash make_plots.sh`

The script `mse.py -i fs/fc` calculates the MSE and Std of SE for the actions of the two intervention methods (fs for full suppression, fc for fixed control). For fixed control, the computation is very intensive and takes around 15 hours.
