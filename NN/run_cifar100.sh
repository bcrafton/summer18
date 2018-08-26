python cifar100_fc.py --epochs 1000 --alpha 0.001 --gpu 0 --dfa 0 --sparse 0 --init sqrt_fan_in --opt adam &
python cifar100_fc.py --epochs 1000 --alpha 0.001 --gpu 1 --dfa 1 --sparse 0 --init zero        --opt adam &
python cifar100_fc.py --epochs 1000 --alpha 0.001 --gpu 2 --dfa 1 --sparse 1 --init zero        --opt adam &

#python cifar100_conv.py --epochs 1000 --alpha 0.00005 --gpu 3 --dfa 0 --sparse 0 --init sqrt_fan_in --opt adam &
#python cifar100_conv.py --epochs 1000 --alpha 0.00005 --gpu 0 --dfa 1 --sparse 0 --init zero        --opt adam &
#python cifar100_conv.py --epochs 1000 --alpha 0.00005 --gpu 1 --dfa 1 --sparse 1 --init zero        --opt adam &
