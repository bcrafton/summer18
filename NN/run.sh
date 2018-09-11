#python mnist_conv.py --epochs 300 --alpha 0.0005 --gpu 0 --dfa 0 --sparse 0 --rank 0 --init sqrt_fan_in --opt adam &
#python mnist_conv.py --epochs 300 --alpha 0.0005 --gpu 1 --dfa 1 --sparse 0 --rank 0 --init zero        --opt adam &
#python mnist_conv.py --epochs 1000 --alpha 0.0005 --gpu 2 --dfa 1 --sparse 1 --rank 0 --init zero        --opt adam &

#python cifar10_conv.py --epochs 500 --alpha 0.00002 --gpu 2 --dfa 0 --sparse 0 --rank 0 --init sqrt_fan_in --opt adam &
#python cifar10_conv.py --epochs 500 --alpha 0.00002 --gpu 3 --dfa 1 --sparse 0 --rank 0 --init zero --opt adam &
#python cifar10_conv.py --epochs 500 --alpha 0.00002 --gpu 1 --dfa 1 --sparse 1 --rank 0 --init zero --opt adam &

#python cifar100_conv.py --epochs 1000 --alpha 0.00005 --gpu 3 --dfa 0 --sparse 0 --rank 0 --init sqrt_fan_in --opt adam &
#python cifar100_conv.py --epochs 1000 --alpha 0.00005 --gpu 0 --dfa 1 --sparse 0 --rank 0 --init zero        --opt adam &
#python cifar100_conv.py --epochs 2000 --alpha 0.00005 --gpu 1 --dfa 1 --sparse 1 --rank 0 --init zero        --opt adam &

#python mnist_fc.py --epochs 500 --alpha 0.005 --gpu 0 --dfa 0 --sparse 0 --rank 0 --init sqrt_fan_in --opt adam &
#python mnist_fc.py --epochs 500 --alpha 0.005 --gpu 1 --dfa 1 --sparse 0 --rank 0 --init zero        --opt adam &
#python mnist_fc.py --epochs 1000 --alpha 0.005 --gpu 2 --dfa 1 --sparse 1 --rank 0 --init zero        --opt adam &

#python cifar10_fc.py --epochs 1000 --alpha 0.00005 --gpu 0 --dfa 0 --sparse 0 --rank 0 --init sqrt_fan_in --opt adam &
#python cifar10_fc.py --epochs 1000 --alpha 0.00005 --gpu 2 --dfa 1 --sparse 0 --rank 0 --init zero        --opt adam &
#python cifar10_fc.py --epochs 1000 --alpha 0.00005 --gpu 3 --dfa 1 --sparse 1 --rank 0 --init zero        --opt adam &

python cifar100_fc.py --epochs 1000 --alpha 0.00001 --gpu 0 --dfa 0 --sparse 0 --rank 0 --init sqrt_fan_in --opt adam &
python cifar100_fc.py --epochs 1000 --alpha 0.00001 --gpu 2 --dfa 1 --sparse 0 --rank 0 --init zero        --opt adam &
python cifar100_fc.py --epochs 1000 --alpha 0.00001 --gpu 3 --dfa 1 --sparse 1 --rank 0 --init zero        --opt adam &
