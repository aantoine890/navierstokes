#!/bin/bash

rm -rf log/log_2SPMV.txt

for i in {1..10}; do
    ./2spmv mat/matrix${i}_aij.mtx >>  log/log_2SPMV.txt 2>&1
done