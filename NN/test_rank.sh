python mnist_fc.py --epochs 200 --batch_size 32 --alpha 0.01 --dfa 1 --sparse 0 --rank 1 --init zero --opt adam --gpu 0 > rank1
python mnist_fc.py --epochs 200 --batch_size 32 --alpha 0.01 --dfa 1 --sparse 0 --rank 2 --init zero --opt adam --gpu 1 > rank2
python mnist_fc.py --epochs 200 --batch_size 32 --alpha 0.01 --dfa 1 --sparse 0 --rank 3 --init zero --opt adam --gpu 2 > rank3
python mnist_fc.py --epochs 200 --batch_size 32 --alpha 0.01 --dfa 1 --sparse 0 --rank 4 --init zero --opt adam --gpu 3 > rank4

python mnist_fc.py --epochs 200 --batch_size 32 --alpha 0.01 --dfa 1 --sparse 0 --rank 5 --init zero --opt adam --gpu 0 > rank5
python mnist_fc.py --epochs 200 --batch_size 32 --alpha 0.01 --dfa 1 --sparse 0 --rank 6 --init zero --opt adam --gpu 1 > rank6
python mnist_fc.py --epochs 200 --batch_size 32 --alpha 0.01 --dfa 1 --sparse 0 --rank 7 --init zero --opt adam --gpu 2 > rank7
python mnist_fc.py --epochs 200 --batch_size 32 --alpha 0.01 --dfa 1 --sparse 0 --rank 8 --init zero --opt adam --gpu 3 > rank8

python mnist_fc.py --epochs 200 --batch_size 32 --alpha 0.01 --dfa 1 --sparse 0 --rank 9 --init zero --opt adam --gpu 0 > rank9
python mnist_fc.py --epochs 200 --batch_size 32 --alpha 0.01 --dfa 1 --sparse 0 --rank 10 --init zero --opt adam --gpu 1 > rank10
