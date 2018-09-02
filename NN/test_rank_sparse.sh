python mnist_fc.py --epochs 200 --batch_size 32 --alpha 0.005 --dfa 1 --sparse 1 --rank 1 --init zero --opt adam --gpu 0 > sparse1rank1 &
python mnist_fc.py --epochs 200 --batch_size 32 --alpha 0.005 --dfa 1 --sparse 1 --rank 2 --init zero --opt adam --gpu 1 > sparse1rank2 &
python mnist_fc.py --epochs 200 --batch_size 32 --alpha 0.005 --dfa 1 --sparse 1 --rank 3 --init zero --opt adam --gpu 2 > sparse1rank3 &
python mnist_fc.py --epochs 200 --batch_size 32 --alpha 0.005 --dfa 1 --sparse 1 --rank 4 --init zero --opt adam --gpu 3 > sparse1rank4 &

wait

python mnist_fc.py --epochs 200 --batch_size 32 --alpha 0.005 --dfa 1 --sparse 1 --rank 5 --init zero --opt adam --gpu 0 > sparse1rank5 &
python mnist_fc.py --epochs 200 --batch_size 32 --alpha 0.005 --dfa 1 --sparse 1 --rank 6 --init zero --opt adam --gpu 1 > sparse1rank6 &
python mnist_fc.py --epochs 200 --batch_size 32 --alpha 0.005 --dfa 1 --sparse 1 --rank 7 --init zero --opt adam --gpu 2 > sparse1rank7 &
python mnist_fc.py --epochs 200 --batch_size 32 --alpha 0.005 --dfa 1 --sparse 1 --rank 8 --init zero --opt adam --gpu 3 > sparse1rank8 &

wait

python mnist_fc.py --epochs 200 --batch_size 32 --alpha 0.005 --dfa 1 --sparse 1 --rank 9 --init zero --opt adam --gpu 0 > sparse1rank9 &
python mnist_fc.py --epochs 200 --batch_size 32 --alpha 0.005 --dfa 1 --sparse 1 --rank 10 --init zero --opt adam --gpu 1 > sparse1rank10 &

#######################################

# python mnist_fc.py --epochs 200 --batch_size 32 --alpha 0.005 --dfa 1 --sparse 2 --rank 1 --init zero --opt adam --gpu 0 > sparse2rank1 &
python mnist_fc.py --epochs 200 --batch_size 32 --alpha 0.005 --dfa 1 --sparse 2 --rank 2 --init zero --opt adam --gpu 1 > sparse2rank2 &
python mnist_fc.py --epochs 200 --batch_size 32 --alpha 0.005 --dfa 1 --sparse 2 --rank 3 --init zero --opt adam --gpu 2 > sparse2rank3 &
python mnist_fc.py --epochs 200 --batch_size 32 --alpha 0.005 --dfa 1 --sparse 2 --rank 4 --init zero --opt adam --gpu 3 > sparse2rank4 &

wait

python mnist_fc.py --epochs 200 --batch_size 32 --alpha 0.005 --dfa 1 --sparse 2 --rank 5 --init zero --opt adam --gpu 0 > sparse2rank5 &
python mnist_fc.py --epochs 200 --batch_size 32 --alpha 0.005 --dfa 1 --sparse 2 --rank 6 --init zero --opt adam --gpu 1 > sparse2rank6 &
python mnist_fc.py --epochs 200 --batch_size 32 --alpha 0.005 --dfa 1 --sparse 2 --rank 7 --init zero --opt adam --gpu 2 > sparse2rank7 &
python mnist_fc.py --epochs 200 --batch_size 32 --alpha 0.005 --dfa 1 --sparse 2 --rank 8 --init zero --opt adam --gpu 3 > sparse2rank8 &

wait

python mnist_fc.py --epochs 200 --batch_size 32 --alpha 0.005 --dfa 1 --sparse 2 --rank 9 --init zero --opt adam --gpu 0 > sparse2rank9 &
python mnist_fc.py --epochs 200 --batch_size 32 --alpha 0.005 --dfa 1 --sparse 2 --rank 10 --init zero --opt adam --gpu 1 > sparse2rank10 &

#######################################

# python mnist_fc.py --epochs 200 --batch_size 32 --alpha 0.005 --dfa 1 --sparse 3 --rank 1 --init zero --opt adam --gpu 0 > sparse3rank1 &
# python mnist_fc.py --epochs 200 --batch_size 32 --alpha 0.005 --dfa 1 --sparse 3 --rank 2 --init zero --opt adam --gpu 1 > sparse3rank2 &
python mnist_fc.py --epochs 200 --batch_size 32 --alpha 0.005 --dfa 1 --sparse 3 --rank 3 --init zero --opt adam --gpu 2 > sparse3rank3 &
python mnist_fc.py --epochs 200 --batch_size 32 --alpha 0.005 --dfa 1 --sparse 3 --rank 4 --init zero --opt adam --gpu 3 > sparse3rank4 &

wait

python mnist_fc.py --epochs 200 --batch_size 32 --alpha 0.005 --dfa 1 --sparse 3 --rank 5 --init zero --opt adam --gpu 0 > sparse3rank5 &
python mnist_fc.py --epochs 200 --batch_size 32 --alpha 0.005 --dfa 1 --sparse 3 --rank 6 --init zero --opt adam --gpu 1 > sparse3rank6 &
python mnist_fc.py --epochs 200 --batch_size 32 --alpha 0.005 --dfa 1 --sparse 3 --rank 7 --init zero --opt adam --gpu 2 > sparse3rank7 &
python mnist_fc.py --epochs 200 --batch_size 32 --alpha 0.005 --dfa 1 --sparse 3 --rank 8 --init zero --opt adam --gpu 3 > sparse3rank8 &

wait

python mnist_fc.py --epochs 200 --batch_size 32 --alpha 0.005 --dfa 1 --sparse 3 --rank 9 --init zero --opt adam --gpu 0 > sparse3rank9 &
python mnist_fc.py --epochs 200 --batch_size 32 --alpha 0.005 --dfa 1 --sparse 3 --rank 10 --init zero --opt adam --gpu 1 > sparse3rank10 &


#######################################

# python mnist_fc.py --epochs 200 --batch_size 32 --alpha 0.005 --dfa 1 --sparse 4 --rank 1 --init zero --opt adam --gpu 0 > sparse4rank1 &
# python mnist_fc.py --epochs 200 --batch_size 32 --alpha 0.005 --dfa 1 --sparse 4 --rank 2 --init zero --opt adam --gpu 1 > sparse4rank2 &
# python mnist_fc.py --epochs 200 --batch_size 32 --alpha 0.005 --dfa 1 --sparse 4 --rank 3 --init zero --opt adam --gpu 2 > sparse4rank3 &
python mnist_fc.py --epochs 200 --batch_size 32 --alpha 0.005 --dfa 1 --sparse 4 --rank 4 --init zero --opt adam --gpu 3 > sparse4rank4 &

wait

python mnist_fc.py --epochs 200 --batch_size 32 --alpha 0.005 --dfa 1 --sparse 4 --rank 5 --init zero --opt adam --gpu 0 > sparse4rank5 &
python mnist_fc.py --epochs 200 --batch_size 32 --alpha 0.005 --dfa 1 --sparse 4 --rank 6 --init zero --opt adam --gpu 1 > sparse4rank6 &
python mnist_fc.py --epochs 200 --batch_size 32 --alpha 0.005 --dfa 1 --sparse 4 --rank 7 --init zero --opt adam --gpu 2 > sparse4rank7 &
python mnist_fc.py --epochs 200 --batch_size 32 --alpha 0.005 --dfa 1 --sparse 4 --rank 8 --init zero --opt adam --gpu 3 > sparse4rank8 &

wait

python mnist_fc.py --epochs 200 --batch_size 32 --alpha 0.005 --dfa 1 --sparse 4 --rank 9 --init zero --opt adam --gpu 0 > sparse4rank9 &
python mnist_fc.py --epochs 200 --batch_size 32 --alpha 0.005 --dfa 1 --sparse 4 --rank 10 --init zero --opt adam --gpu 1 > sparse4rank10 &

#######################################

# python mnist_fc.py --epochs 200 --batch_size 32 --alpha 0.005 --dfa 1 --sparse 5 --rank 1 --init zero --opt adam --gpu 0 > sparse5rank1 &
# python mnist_fc.py --epochs 200 --batch_size 32 --alpha 0.005 --dfa 1 --sparse 5 --rank 2 --init zero --opt adam --gpu 1 > sparse5rank2 &
# python mnist_fc.py --epochs 200 --batch_size 32 --alpha 0.005 --dfa 1 --sparse 5 --rank 3 --init zero --opt adam --gpu 2 > sparse5rank3 &
# python mnist_fc.py --epochs 200 --batch_size 32 --alpha 0.005 --dfa 1 --sparse 5 --rank 4 --init zero --opt adam --gpu 3 > sparse5rank4 &

python mnist_fc.py --epochs 200 --batch_size 32 --alpha 0.005 --dfa 1 --sparse 5 --rank 5 --init zero --opt adam --gpu 0 > sparse5rank5 &
python mnist_fc.py --epochs 200 --batch_size 32 --alpha 0.005 --dfa 1 --sparse 5 --rank 6 --init zero --opt adam --gpu 1 > sparse5rank6 &
python mnist_fc.py --epochs 200 --batch_size 32 --alpha 0.005 --dfa 1 --sparse 5 --rank 7 --init zero --opt adam --gpu 2 > sparse5rank7 &
python mnist_fc.py --epochs 200 --batch_size 32 --alpha 0.005 --dfa 1 --sparse 5 --rank 8 --init zero --opt adam --gpu 3 > sparse5rank8 &

wait

python mnist_fc.py --epochs 200 --batch_size 32 --alpha 0.005 --dfa 1 --sparse 5 --rank 9 --init zero --opt adam --gpu 0 > sparse5rank9 &
python mnist_fc.py --epochs 200 --batch_size 32 --alpha 0.005 --dfa 1 --sparse 5 --rank 10 --init zero --opt adam --gpu 1 > sparse5rank10 &

#######################################

# python mnist_fc.py --epochs 200 --batch_size 32 --alpha 0.005 --dfa 1 --sparse 6 --rank 1 --init zero --opt adam --gpu 0 > sparse6rank1 &
# python mnist_fc.py --epochs 200 --batch_size 32 --alpha 0.005 --dfa 1 --sparse 6 --rank 2 --init zero --opt adam --gpu 1 > sparse6rank2 &
# python mnist_fc.py --epochs 200 --batch_size 32 --alpha 0.005 --dfa 1 --sparse 6 --rank 3 --init zero --opt adam --gpu 2 > sparse6rank3 &
# python mnist_fc.py --epochs 200 --batch_size 32 --alpha 0.005 --dfa 1 --sparse 6 --rank 4 --init zero --opt adam --gpu 3 > sparse6rank4 &

# python mnist_fc.py --epochs 200 --batch_size 32 --alpha 0.005 --dfa 1 --sparse 6 --rank 5 --init zero --opt adam --gpu 0 > sparse6rank5 &
python mnist_fc.py --epochs 200 --batch_size 32 --alpha 0.005 --dfa 1 --sparse 6 --rank 6 --init zero --opt adam --gpu 1 > sparse6rank6 &
python mnist_fc.py --epochs 200 --batch_size 32 --alpha 0.005 --dfa 1 --sparse 6 --rank 7 --init zero --opt adam --gpu 2 > sparse6rank7 &
python mnist_fc.py --epochs 200 --batch_size 32 --alpha 0.005 --dfa 1 --sparse 6 --rank 8 --init zero --opt adam --gpu 3 > sparse6rank8 &

wait

python mnist_fc.py --epochs 200 --batch_size 32 --alpha 0.005 --dfa 1 --sparse 6 --rank 9 --init zero --opt adam --gpu 0 > sparse6rank9 &
python mnist_fc.py --epochs 200 --batch_size 32 --alpha 0.005 --dfa 1 --sparse 6 --rank 10 --init zero --opt adam --gpu 1 > sparse6rank10 &

#######################################

# python mnist_fc.py --epochs 200 --batch_size 32 --alpha 0.005 --dfa 1 --sparse 7 --rank 1 --init zero --opt adam --gpu 0 > sparse7rank1 &
# python mnist_fc.py --epochs 200 --batch_size 32 --alpha 0.005 --dfa 1 --sparse 7 --rank 2 --init zero --opt adam --gpu 1 > sparse7rank2 &
# python mnist_fc.py --epochs 200 --batch_size 32 --alpha 0.005 --dfa 1 --sparse 7 --rank 3 --init zero --opt adam --gpu 2 > sparse7rank3 &
# python mnist_fc.py --epochs 200 --batch_size 32 --alpha 0.005 --dfa 1 --sparse 7 --rank 4 --init zero --opt adam --gpu 3 > sparse7rank4 &

# python mnist_fc.py --epochs 200 --batch_size 32 --alpha 0.005 --dfa 1 --sparse 7 --rank 5 --init zero --opt adam --gpu 0 > sparse7rank5 &
# python mnist_fc.py --epochs 200 --batch_size 32 --alpha 0.005 --dfa 1 --sparse 7 --rank 6 --init zero --opt adam --gpu 1 > sparse7rank6 &
python mnist_fc.py --epochs 200 --batch_size 32 --alpha 0.005 --dfa 1 --sparse 7 --rank 7 --init zero --opt adam --gpu 2 > sparse7rank7 &
python mnist_fc.py --epochs 200 --batch_size 32 --alpha 0.005 --dfa 1 --sparse 7 --rank 8 --init zero --opt adam --gpu 3 > sparse7rank8 &

wait

python mnist_fc.py --epochs 200 --batch_size 32 --alpha 0.005 --dfa 1 --sparse 7 --rank 9 --init zero --opt adam --gpu 0 > sparse7rank9 &
python mnist_fc.py --epochs 200 --batch_size 32 --alpha 0.005 --dfa 1 --sparse 7 --rank 10 --init zero --opt adam --gpu 1 > sparse7rank10 &


#######################################

# python mnist_fc.py --epochs 200 --batch_size 32 --alpha 0.005 --dfa 1 --sparse 8 --rank 1 --init zero --opt adam --gpu 0 > sparse8rank1 &
# python mnist_fc.py --epochs 200 --batch_size 32 --alpha 0.005 --dfa 1 --sparse 8 --rank 2 --init zero --opt adam --gpu 1 > sparse8rank2 &
# python mnist_fc.py --epochs 200 --batch_size 32 --alpha 0.005 --dfa 1 --sparse 8 --rank 3 --init zero --opt adam --gpu 2 > sparse8rank3 &
# python mnist_fc.py --epochs 200 --batch_size 32 --alpha 0.005 --dfa 1 --sparse 8 --rank 4 --init zero --opt adam --gpu 3 > sparse8rank4 &

# python mnist_fc.py --epochs 200 --batch_size 32 --alpha 0.005 --dfa 1 --sparse 8 --rank 5 --init zero --opt adam --gpu 0 > sparse8rank5 &
# python mnist_fc.py --epochs 200 --batch_size 32 --alpha 0.005 --dfa 1 --sparse 8 --rank 6 --init zero --opt adam --gpu 1 > sparse8rank6 &
# python mnist_fc.py --epochs 200 --batch_size 32 --alpha 0.005 --dfa 1 --sparse 8 --rank 7 --init zero --opt adam --gpu 2 > sparse8rank7 &
python mnist_fc.py --epochs 200 --batch_size 32 --alpha 0.005 --dfa 1 --sparse 8 --rank 8 --init zero --opt adam --gpu 3 > sparse8rank8 &

python mnist_fc.py --epochs 200 --batch_size 32 --alpha 0.005 --dfa 1 --sparse 8 --rank 9 --init zero --opt adam --gpu 0 > sparse8rank9 &
python mnist_fc.py --epochs 200 --batch_size 32 --alpha 0.005 --dfa 1 --sparse 8 --rank 10 --init zero --opt adam --gpu 1 > sparse8rank10 &

wait

#######################################

# python mnist_fc.py --epochs 200 --batch_size 32 --alpha 0.005 --dfa 1 --sparse 9 --rank 1 --init zero --opt adam --gpu 0 > sparse9rank1 &
# python mnist_fc.py --epochs 200 --batch_size 32 --alpha 0.005 --dfa 1 --sparse 9 --rank 2 --init zero --opt adam --gpu 1 > sparse9rank2 &
# python mnist_fc.py --epochs 200 --batch_size 32 --alpha 0.005 --dfa 1 --sparse 9 --rank 3 --init zero --opt adam --gpu 2 > sparse9rank3 &
# python mnist_fc.py --epochs 200 --batch_size 32 --alpha 0.005 --dfa 1 --sparse 9 --rank 4 --init zero --opt adam --gpu 3 > sparse9rank4 &

# python mnist_fc.py --epochs 200 --batch_size 32 --alpha 0.005 --dfa 1 --sparse 9 --rank 5 --init zero --opt adam --gpu 0 > sparse9rank5 &
# python mnist_fc.py --epochs 200 --batch_size 32 --alpha 0.005 --dfa 1 --sparse 9 --rank 6 --init zero --opt adam --gpu 1 > sparse9rank6 &
# python mnist_fc.py --epochs 200 --batch_size 32 --alpha 0.005 --dfa 1 --sparse 9 --rank 7 --init zero --opt adam --gpu 2 > sparse9rank7 &
# python mnist_fc.py --epochs 200 --batch_size 32 --alpha 0.005 --dfa 1 --sparse 9 --rank 8 --init zero --opt adam --gpu 3 > sparse9rank8 &

python mnist_fc.py --epochs 200 --batch_size 32 --alpha 0.005 --dfa 1 --sparse 9 --rank 9 --init zero --opt adam --gpu 0 > sparse9rank9 &
python mnist_fc.py --epochs 200 --batch_size 32 --alpha 0.005 --dfa 1 --sparse 9 --rank 10 --init zero --opt adam --gpu 1 > sparse9rank10 &

#######################################

# python mnist_fc.py --epochs 200 --batch_size 32 --alpha 0.005 --dfa 1 --sparse 10 --rank 1 --init zero --opt adam --gpu 0 > sparse10rank1 &
# python mnist_fc.py --epochs 200 --batch_size 32 --alpha 0.005 --dfa 1 --sparse 10 --rank 2 --init zero --opt adam --gpu 1 > sparse10rank2 &
# python mnist_fc.py --epochs 200 --batch_size 32 --alpha 0.005 --dfa 1 --sparse 10 --rank 3 --init zero --opt adam --gpu 2 > sparse10rank3 &
# python mnist_fc.py --epochs 200 --batch_size 32 --alpha 0.005 --dfa 1 --sparse 10 --rank 4 --init zero --opt adam --gpu 3 > sparse10rank4 &

# python mnist_fc.py --epochs 200 --batch_size 32 --alpha 0.005 --dfa 1 --sparse 10 --rank 5 --init zero --opt adam --gpu 0 > sparse10rank5 &
# python mnist_fc.py --epochs 200 --batch_size 32 --alpha 0.005 --dfa 1 --sparse 10 --rank 6 --init zero --opt adam --gpu 1 > sparse10rank6 &
# python mnist_fc.py --epochs 200 --batch_size 32 --alpha 0.005 --dfa 1 --sparse 10 --rank 7 --init zero --opt adam --gpu 2 > sparse10rank7 &
# python mnist_fc.py --epochs 200 --batch_size 32 --alpha 0.005 --dfa 1 --sparse 10 --rank 8 --init zero --opt adam --gpu 3 > sparse10rank8 &

# python mnist_fc.py --epochs 200 --batch_size 32 --alpha 0.005 --dfa 1 --sparse 10 --rank 9 --init zero --opt adam --gpu 0 > sparse10rank9 &
python mnist_fc.py --epochs 200 --batch_size 32 --alpha 0.005 --dfa 1 --sparse 10 --rank 10 --init zero --opt adam --gpu 1 > sparse10rank10 &
