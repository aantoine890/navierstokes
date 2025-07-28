#!/bin/bash

rm -rf log/log_SPMV.txt

for i in {1..10}; do
    ./spmv mat/matrix${i}_aij.mtx >>  log/log_SPMV.txt 2>&1
done