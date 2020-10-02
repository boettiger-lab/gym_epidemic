#!/bin/bash

if test -f "fs_intervention.png"; then
    rm fs_intervention.png
fi
python plot_png.py
if test -f "fs_intervention.png"; then
    echo "fs_intervention.png has been created."
fi


if test -f "fc_intervention.png"; then
    rm fc_intervention.png
fi
if test -f "fc_sigma.png"; then
    rm fc_sigma.png
fi
python plot_png.py -i fc
if test -f "fc_intervention.png"; then
    echo "fc_intervention.png has been created."
fi
if test -f "fc_sigma.png"; then
    echo "fc_sigma.png has been created."
fi


if test -f "reward_plot_2d_fs.png"; then
    rm reward_plot_2d_fs.png
fi
python reward_plot_2d_fs.py
if test -f "reward_plot_2d_fs.png"; then
    echo "reward_plot_2d_fs.png has been created."
fi


if test -f "reward_plot_2d_fc.png"; then
    rm reward_plot_2d_fc.png
fi
python reward_plot_2d_fc.py
if test -f "reward_plot_2d_fc.png"; then
    echo "reward_plot_2d_fc.png has been created."
fi

