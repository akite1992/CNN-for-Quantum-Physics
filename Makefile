## Set which compiler to use by defining CCCOM:
##GNU GCC compiler
CC=g++  -std=c++11
#FC=gfortran-4.9
LIB=-larmadillo

##Clang compiler (good to use on Mac OS)
#CCCOM=clang++ -std=c++1y
##Intel C++ compiler (good to use with Intel MKL if available)
##CC=icpc -std=c++11 -gxx-name=g++-4.9 
#########


OBJS = main.o 
DEBUG = -g
#CFLAGS = -c $(DEBUG)
#LFLAGS = $(DEBUG)
CFLAGS = -O2 -Wall -c $(DEBUG)
LFLAGS = -Wall $(DEBUG)

neural: $(OBJS)
	$(CC)  $(OBJS) $(LIB) -o ./neural

clean:
	rm *.o *~

entanglement.o: main.cpp neural_network.h conv_set.h final_layer.h fc_layer.h conv_layer.h pool_layer.h nonlinear_functions.h 
	$(CC) $(CFLAGS) main.cpp 

