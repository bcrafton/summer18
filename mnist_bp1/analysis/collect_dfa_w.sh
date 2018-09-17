#!/bin/bash
for i in {1..250}
do
   python mnist_dfa.py --gpu 0 --num $i --load B.npy            &
   python mnist_dfa.py --gpu 1 --num $i --load BAD_B.npy        &
   python mnist_dfa.py --gpu 2 --num $i --load SPARSE_B.npy     &
   python mnist_dfa.py --gpu 3 --num $i --load SPARSE_BAD_B.npy &
   wait
done
