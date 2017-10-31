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

const int full_size=5;

class fully_connected_layer{
public:
	mat weight;
	mat grad_weight;
	mat bias;
	mat grad_bias;
	mat node;
	mat delta_node;
	mat update_value;
	int update_num;
	int update_start;
	int update_end;
	int size;
	int prenum;
	int presize;
	int num_par;
	int num_start;
	int num_end;
	fully_connected_layer(int prenum, int presize ,int y, int para_start);
	void forward(mat & prenodes);
	void forward_onesite(int num, int start, int end, mat & value);
	void update();
	void back_prop(mat & pre_nodes,mat & delta_pre_nodes, vec & update_Grads);
	void save_parameter(vec & A);
	void load_parameter(vec & A);
	void update_parameter(vec & A);
};

fully_connected_layer::fully_connected_layer(int pprenum, int ppresize ,int y, int para_start){
	size=y;
	prenum=pprenum;
	presize=ppresize;
	node.set_size(1,size);
	update_value.set_size(1,size);
	weight.set_size(prenum*presize,size);
	grad_weight.set_size(prenum*presize,size);
	delta_node.set_size(1,size);
	bias.set_size(1,size);
	grad_bias.set_size(1,size);
	bias.fill(0);
	weight.randu();
	double a=sqrt(6)/sqrt(prenum*presize+size);
	weight=weight*2*a-a;
	num_start=para_start;
	num_par=prenum*presize*size+size;
	num_end=num_start+num_par-1;
}

void fully_connected_layer::forward(mat & prenodes){
	int in;
	int i;
	int j;
	for(j=0;j<size;j++){
		node[j]=0;
		for(in=0;in<prenum;in++){
			for(i=0;i<presize;i++){
				node[j]+=prenodes.at(in,i)*weight.at(in*presize+i,j);
			}
		}
		node[j]+=bias[j];
	}
	//output/=innum*size;

}

void fully_connected_layer::forward_onesite(int num, int start, int end, mat & value){
	update_num=size;
	update_start=0;
	update_end=size-1;
	int in;
	int i,j;
	int index;
	for(j=0;j<size;j++){
		update_value[j]=0;
		for(in=0;in<prenum;in++){
			for(i=0;i<num;i++){
				index=ZL(i+start,presize);
				update_value[j]+=weight.at(in*presize+index,j)*value.at(in,index);
			}
		}
	}
}

void fully_connected_layer::update(){
	node+=update_value;
}

void fully_connected_layer::back_prop(mat & pre_nodes,mat & delta_pre_nodes, vec & update_Grads){

	grad_weight.fill(0);
	delta_pre_nodes.fill(0);
	grad_bias=delta_node;
	int in,i,j;
	for(j=0;j<size;j++){
		for(in=0;in<prenum;in++){
			for(i=0;i<presize;i++){
				delta_pre_nodes.at(in,i)+=delta_node[j]*weight.at(in*presize+i,j);
				grad_weight.at(in*presize+i,j)+=delta_node[j]*pre_nodes.at(in,i);
			}
		}
	}

	for(i=0;i<prenum*presize*size;i++){
		update_Grads[i+num_start]=grad_weight[i];
	}
	for(i=0;i<size;i++){
		update_Grads[prenum*presize*size+i+num_start]=grad_bias[i];
	}
}

void fully_connected_layer::save_parameter(vec & A){
	int i;
	for(i=0;i<prenum*presize*size;i++){
		A[i+num_start]=weight[i];
	}
	for(i=0;i<size;i++){
		A[prenum*presize*size+i+num_start]=bias[i];
	}
	
}

void fully_connected_layer::load_parameter(vec & A){
	int i;
	for(i=0;i<prenum*presize*size;i++){
		weight[i]=A[i+num_start];
	}
	for(i=0;i<size;i++){
		bias[i]=A[prenum*presize*size+i+num_start];
	}
	
}

void fully_connected_layer::update_parameter(vec & A){
	int i;
	for(i=0;i<prenum*presize*size;i++){
		weight[i]+=A[i+num_start];
	}
	for(i=0;i<size;i++){
		bias[i]+=A[prenum*presize*size+i+num_start];
	}
	
}