#!/bin/bash

rm -rf log/log_SPM2V.txt

for i in {1..10}; do
    ./spm2v mat/matrix${i}_aij.mtx >>  log/log_SPM2V.txt 2>&1
done