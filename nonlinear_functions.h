#ifndef Nonlinear_H
#define Nonlinear_H

#include <iostream>
#include <cmath>
using namespace std;

double ZL(int i, int NN){ //module
	int a=i%NN;
	if(a<0){
		a+=NN;
	}
	return a;
}
double softplus(double x){
	return log(1+exp(x));
}

double sigmoid( double x){
	return 1./(1+exp(-x));
}

double grad_sigmoid(double x){
	double a=exp(-x);
	return a/pow(1+a,2);
}

inline double Relu(double x){
	double a=x;
	if(a<0){a=0;}
	return a;
}

inline double grad_Relu(double x){
	double a=1;
	if(x<0){a=0;}
	return a;
}
inline double grad_tanh(double x){
	return 1/pow(cosh(x),2);
}

inline double grad_softplus(double x){
	if(x<-10){
		return 0;
	}
	else{
		return 1./(1+exp(-x))/log(1+exp(x));
	}
}
#endif