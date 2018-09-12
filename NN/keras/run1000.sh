#!/bin/bash
for k in {484..1000}
do
    python keras_mnist_fc.py --num $k --epochs 20 --alpha 0.01 --gpu 0 --verbose 0
done

