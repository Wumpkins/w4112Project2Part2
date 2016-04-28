all:
	gcc -std=c99 -pedantic -O3 -march=native -msse4.2 -msse4a -o build main.c p2random.c tree.c
