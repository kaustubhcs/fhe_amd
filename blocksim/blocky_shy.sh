#!/bin/bash

# This script is used to run the blocky simulation with different parameters

EXP_NAME=V006_TR_10

LDS_CAPACITY=7.5
MAX_LDS_CAPACITY=20.5
LDS_CAPACITY_STEP=1

START_TEST=1
TOTAL_TESTS=12
# TOTAL_TESTS=1


# Run the simulation for different LDS capacities
for lds in `seq $LDS_CAPACITY $LDS_CAPACITY_STEP $MAX_LDS_CAPACITY`;
do
    for (( i=$START_TEST; i<$TOTAL_TESTS; i++ ))
    do
        echo "Running test $i"
        echo "LDS capacity: $lds"
        python ./2blocky.py $EXP_NAME $lds $i &
        # LDS_CAPACITY=$(echo "$LDS_CAPACITY + $LDS_CAPACITY_STEP" | bc)
    done
done
