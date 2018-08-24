python Cifar10_large.py --epochs 500 --alpha 0.00002 --gpu 0 --dfa 0 --sparse 0 --init sqrt_fan_in --opt adam &
python Cifar10_large.py --epochs 500 --alpha 0.00002 --gpu 1 --dfa 1 --sparse 0 --init zero        --opt adam &
python Cifar10_large.py --epochs 500 --alpha 0.00002 --gpu 2 --dfa 1 --sparse 1 --init zero        --opt adam &

python Cifar100.py      --epochs 500 --alpha 0.00001 --gpu 3 --dfa 0 --sparse 0 --init sqrt_fan_in --opt adam &
python Cifar100.py      --epochs 500 --alpha 0.00001 --gpu 0 --dfa 1 --sparse 0 --init zero        --opt adam &
python Cifar100.py      --epochs 500 --alpha 0.00001 --gpu 1 --dfa 1 --sparse 1 --init zero        --opt adam &
