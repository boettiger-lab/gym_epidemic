#!/bin/bash

python plot_png.py
echo "fs_intervention.png has been created."
python plot_png.py -i fc
echo "fc_intervention.png and fc_sigma.png have been created."
python reward_plot_2d_fs.py
echo "reward_plot_2d_fs.png has been created."
python reward_plot_2d_fc.py
echo "reward_plot_2d_fc.png has been created."


