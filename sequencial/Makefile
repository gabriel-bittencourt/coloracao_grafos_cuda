MAKEFLAGS += --no-print-directory

CC = g++
CFLAGS = -Wall -g

Graph.o: Graph.h
Coloring.o: Coloring.h

main.o:  main.cpp Graph.o Coloring.o
	$(CC) $(CFLAGS) -c main.cpp

main: main.o Graph.o Coloring.o
	$(CC) $(CFLAGS) -o main main.o Graph.o Coloring.o

run: main
	./main

clean:
	rm -f *.o main
