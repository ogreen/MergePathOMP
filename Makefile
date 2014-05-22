all:
	gcc -fopenmp -std=c99 -O3 *.c

testall: test1m test10m test100m testweak

test1m:
	./a.out -A 1000000 -B 1000000 -t 2
	./a.out -A 1000000 -B 1000000 -t 4
	./a.out -A 1000000 -B 1000000 -t 8
	./a.out -A 1000000 -B 1000000 -t 16
	
test10m:
	./a.out -A 10000000 -B 10000000 -t 2
	./a.out -A 10000000 -B 10000000 -t 4
	./a.out -A 10000000 -B 10000000 -t 8
	./a.out -A 10000000 -B 10000000 -t 16

test100m:
	./a.out -A 100000000 -B 100000000 -t 2
	./a.out -A 100000000 -B 100000000 -t 4
	./a.out -A 100000000 -B 100000000 -t 8
	./a.out -A 100000000 -B 100000000 -t 16

testweak:
	./a.out -A 2000000 -B 2000000 -t 2
	./a.out -A 4000000 -B 4000000 -t 4
	./a.out -A 8000000 -B 8000000 -t 8
	./a.out -A 16000000 -B 16000000 -t 16
