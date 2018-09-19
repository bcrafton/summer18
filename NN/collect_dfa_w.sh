#!/bin/bash
for i in {1..250}
do
   #python mnist_fc.py --epochs 200 --batch_size 32 --alpha 0.01 --gpu 0 --dfa 1 --sparse 0 --rank 0 --init sqrt_fan_in --opt adam --save 1 --num $i --load ./analysis/random_feedback/B.npy &
   #python mnist_fc.py --epochs 200 --batch_size 32 --alpha 0.01 --gpu 1 --dfa 1 --sparse 0 --rank 0 --init sqrt_fan_in --opt adam --save 1 --num $i --load ./analysis/random_feedback/BAD_B.npy &
   #python mnist_fc.py --epochs 200 --batch_size 32 --alpha 0.01 --gpu 2 --dfa 1 --sparse 0 --rank 0 --init sqrt_fan_in --opt adam --save 1 --num $i --load ./analysis/random_feedback/SPARSE_B.npy &
   #python mnist_fc.py --epochs 200 --batch_size 32 --alpha 0.01 --gpu 3 --dfa 1 --sparse 0 --rank 0 --init sqrt_fan_in --opt adam --save 1 --num $i --load ./analysis/random_feedback/SPARSE_BAD_B.npy &
   
   python mnist_fc.py --epochs 200 --batch_size 32 --alpha 0.01 --gpu 0 --dfa 1 --sparse 0 --rank 0 --init zero --opt adam --save 1 --num $i &
   python mnist_fc.py --epochs 200 --batch_size 32 --alpha 0.01 --gpu 1 --dfa 1 --sparse 0 --rank 1 --init zero --opt adam --save 1 --num $i &
   python mnist_fc.py --epochs 200 --batch_size 32 --alpha 0.01 --gpu 2 --dfa 1 --sparse 1 --rank 0 --init zero --opt adam --save 1 --num $i &
   python mnist_fc.py --epochs 200 --batch_size 32 --alpha 0.01 --gpu 3 --dfa 1 --sparse 1 --rank 1 --init zero --opt adam --save 1 --num $i &

   wait
done
