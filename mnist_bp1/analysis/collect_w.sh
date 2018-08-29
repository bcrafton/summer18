#!/bin/bash
for i in {1..2}
do
   python mnist.py --gpu 0 --num $i &
   python mnist.py --gpu 1 --num $i &
   python mnist.py --gpu 2 --num $i &
   python mnist.py --gpu 3 --num $i &
   wait
done
