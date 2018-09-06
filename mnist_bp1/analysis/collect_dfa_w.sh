#!/bin/bash
for i in {1..2500}
do
   python mnist_dfa.py --gpu 0 --num $i &
   python mnist_dfa.py --gpu 1 --num $i &
   python mnist_dfa.py --gpu 2 --num $i &
   python mnist_dfa.py --gpu 3 --num $i &
   wait
done
