#include <iostream>
#include <vector>
#include <cmath>
#include <ctime>
#include <fstream>
#include <random>
#include "time.h"
#include <armadillo>
#include <iomanip>
#include "nonlinear_functions.h"
using namespace std;
using namespace arma;

class final_layer{
public:
	double output;
	mat weight;
	double bias;
	double update_output;
	double delta_output; // gradient of output
	mat grad_weight;
	double grad_bias;
	int innum; //number of images before final layer
	int size; //size of image before final layer
	int num_par;
	int num_start;
	int num_end;
	string name;
	void save_parameter(vec & A);
	void load_parameter(vec & A);
	void update_parameter(vec &A);
	final_layer(int x,int y, int para_start);
	void forward(mat & nodes);
	void forward_onesite(int num, int start, int end, mat & vlaue );
	void update();
	void back_prop(mat &pre_nodes,mat &delta_pre_nodes, vec & update_Grads); // here delta is the delta of node in the last layer.
};

final_layer::final_layer(int x,int y, int para_start){
	innum=x;
	size=y;
	weight.set_size(innum,size);
	grad_weight.set_size(innum,size);
	bias=0;
	weight.randu();
	double a=sqrt(6)/sqrt(innum*size+1);
	weight=weight*2*a-a;
	num_start=para_start;
	num_par=innum*size+1;
	num_end=num_start+num_par-1;
	//weight.fill(0.1);
}

void final_layer::forward(mat & nodes){
	int in;
	int i;
	output=0;
	for(in=0;in<innum;in++){
		for(i=0;i<size;i++){
			output+=nodes.at(in,i)*weight.at(in,i);
		}
	}
	output+=bias;
	//output/=innum*size;

}

void final_layer::forward_onesite(int num, int start, int end, mat & value){
	update_output=0;
	int in;
	int i;
	int index;
	for(in=0;in<innum;in++){
		for(i=0;i<num;i++){
			index=ZL((i+start),size);
			update_output+=weight.at(in,index)*value.at(in,index);
		}
	}
}

void final_layer::back_prop(mat & pre_nodes,mat & delta_pre_nodes, vec & update_Grads){
	grad_bias=delta_output;
	grad_weight=pre_nodes*delta_output;

	
	delta_pre_nodes=weight*delta_output;
	int i;
	for(i=0;i<innum*size;i++){
		update_Grads[i+num_start]=grad_weight[i];
	}
	update_Grads[innum*size+num_start]=grad_bias;
	
}


void final_layer::update(){
	output+=update_output;
}

void final_layer::load_parameter(vec & A){
	int i;
	for(i=0;i<innum*size;i++){
		weight[i]=A[i+num_start];
	}
	bias=A[innum*size+num_start];
}

void final_layer::update_parameter(vec & A){
	int i;
	for(i=0;i<innum*size;i++){
		weight[i]+=A[i+num_start];
	}
	bias+=A[innum*size+num_start];
}
void final_layer::save_parameter(vec & A){
	int i;
	for(i=0;i<innum*size;i++){
		A[i+num_start]=weight[i];
	}
	A[innum*size+num_start]=bias;
}