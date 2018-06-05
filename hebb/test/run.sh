
######

python hebb.py --iters 1 --examples 10000 --hi 10.0 --lo 0.01 --lr 0.0001 >> results
python hebb.py --iters 3 --examples 10000 --hi 10.0 --lo 0.01 --lr 0.0001 >> results
python hebb.py --iters 5 --examples 10000 --hi 10.0 --lo 0.01 --lr 0.0001 >> results

python hebb.py --iters 1 --examples 10000 --hi 10.0 --lo 0.01 --lr 0.01 >> results
python hebb.py --iters 3 --examples 10000 --hi 10.0 --lo 0.01 --lr 0.01 >> results
python hebb.py --iters 5 --examples 10000 --hi 10.0 --lo 0.01 --lr 0.01 >> results

python hebb.py --iters 1 --examples 10000 --hi 10.0 --lo 0.01 --lr 0.001 >> results
python hebb.py --iters 3 --examples 10000 --hi 10.0 --lo 0.01 --lr 0.001 >> results
python hebb.py --iters 5 --examples 10000 --hi 10.0 --lo 0.01 --lr 0.001 >> results

python hebb.py --iters 1 --examples 10000 --hi 100.0 --lo 0.001 --lr 0.001 >> results
python hebb.py --iters 3 --examples 10000 --hi 100.0 --lo 0.001 --lr 0.001 >> results
python hebb.py --iters 3 --examples 10000 --hi 100.0 --lo 0.001 --lr 0.001 >> results

######

python hebb2layer.py --iters 1 --examples 10000 --hi 10.0 --lo 0.01 --lr 0.0001 >> results
python hebb2layer.py --iters 3 --examples 10000 --hi 10.0 --lo 0.01 --lr 0.0001 >> results
python hebb2layer.py --iters 5 --examples 10000 --hi 10.0 --lo 0.01 --lr 0.0001 >> results

python hebb2layer.py --iters 1 --examples 10000 --hi 10.0 --lo 0.01 --lr 0.01 >> results
python hebb2layer.py --iters 3 --examples 10000 --hi 10.0 --lo 0.01 --lr 0.01 >> results
python hebb2layer.py --iters 5 --examples 10000 --hi 10.0 --lo 0.01 --lr 0.01 >> results

python hebb2layer.py --iters 1 --examples 10000 --hi 10.0 --lo 0.01 --lr 0.001 >> results
python hebb2layer.py --iters 3 --examples 10000 --hi 10.0 --lo 0.01 --lr 0.001 >> results
python hebb2layer.py --iters 5 --examples 10000 --hi 10.0 --lo 0.01 --lr 0.001 >> results

python hebb2layer.py --iters 1 --examples 10000 --hi 100.0 --lo 0.001 --lr 0.001 >> results
python hebb2layer.py --iters 3 --examples 10000 --hi 100.0 --lo 0.001 --lr 0.001 >> results
python hebb2layer.py --iters 3 --examples 10000 --hi 100.0 --lo 0.001 --lr 0.001 >> results
