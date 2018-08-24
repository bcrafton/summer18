python mnist_fc.py --epochs 100 --alpha 0.0001 --gpu 0 --dfa 0 --sparse 0 --init sqrt_fan_in --opt adam &
python mnist_fc.py --epochs 100 --alpha 0.0001 --gpu 1 --dfa 1 --sparse 0 --init zero        --opt adam &
python mnist_fc.py --epochs 100 --alpha 0.0001 --gpu 2 --dfa 1 --sparse 1 --init zero        --opt adam &

python mnist_conv.py --epochs 100 --alpha 0.0001 --gpu 3 --dfa 0 --sparse 0 --init sqrt_fan_in --opt adam &
python mnist_conv.py --epochs 100 --alpha 0.0001 --gpu 0 --dfa 1 --sparse 0 --init zero        --opt adam &
python mnist_conv.py --epochs 100 --alpha 0.0001 --gpu 1 --dfa 1 --sparse 1 --init zero        --opt adam &
