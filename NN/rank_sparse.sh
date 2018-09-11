#!/bin/bash
for i in {1..10}
do
    for (( j=$i; j<=10; j++ ))
    do
        for k in {1..10}
        do
            fname=sparse$i
            fname+=rank$j
            fname+=Itr$k
            # echo $fname
            
            gpu=$(($k%4))
            # echo $gpu
            
            # echo python mnist_fc.py --epochs 500 --batch_size 32 --alpha 0.005 --dfa 1 --sparse $i --rank $j --init zero --opt adam --gpu $gpu --name $fname &
            python mnist_fc.py --epochs 500 --batch_size 32 --alpha 0.005 --dfa 1 --sparse $i --rank $j --init zero --opt adam --gpu $gpu --name $fname &
        done
        wait
    done
done
