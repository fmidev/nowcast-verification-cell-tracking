#!/bin/bash

export PYTHONPATH=$PYTHONPATH:.

SCRIPT=scripts/plot_nowcast_figures.py
CONFIG=config/plot_nowcast_figs.yml

declare -a DATES=(
    "202105100815"
    "202107242015"
    "202205161615"
    "202105101015"
    "202206050120"
    "202107140920"
    "202107140810"
    "202107060500"
    "202108011340"
    "202207010955"
    "202208260250"
    "202209272235"
    "202108011435"
    "202206281510"
)

for date in "${DATES[@]}"; do
    python "$SCRIPT" "$CONFIG" "$date" # || exit 1
done
