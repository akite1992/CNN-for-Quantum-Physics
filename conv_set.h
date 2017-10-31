#include <iostream>
#include <vector>
#include <cmath>
#include <ctime>
#include <fstream>
#include <random>
#include "time.h"
#include <armadillo>
#include <iomanip>
#include "conv_layer.h"
#include "pool_layer.h"
using namespace std;
using namespace arma;
const int layer_num=1;
class conv_set{  // several conv_layers
public:
	conv_layer *  clayer[layer_num];
	pooling_layer * player;
	int size;
	int num_start;
	int num_end;
	int num_par;
	int innum;
	int outnum;
	int kernel_size;
	conv_set(int x, int y, int z, int kernel, int para_start);
	void forward(mat & spin);
	void forward_onesite(int num, int start, int end, mat & value);
	void update();
	void back_prop(mat & pre_nodes,mat & delta_pre_nodes, vec & update_Grads);
	void first_back_prop(mat & pre_nodes, vec & update_Grads);
	void save_parameter(vec & A);
	void load_parameter(vec & A);
	void update_parameter(vec & A);
};

conv_set::conv_set(int x, int y, int z, int kernel, int para_start){
	size=z;
	innum=x;
	outnum=y;
	kernel_size=kernel;
	int i;
	num_start=para_start;
	int num_index=num_start;
	num_par=0;
	for(i=0;i<layer_num;i++){
		clayer[i]=new conv_layer(innum, outnum,size, kernel_size,num_index);
		num_index=clayer[i]->num_end+1;
		num_par+=clayer[i]->num_par;
	}
	player=new pooling_layer(outnum, size, pool_size);
	num_end=num_start+num_par-1;
}

void conv_set::forward(mat & spin){
	int i;
	clayer[0]->forward(spin);
	for(i=1;i<layer_num;i++){
		clayer[i]->forward(clayer[i-1]->node_act);
	}
	player->forward(clayer[layer_num-1]->node_act, size);
}

void conv_set::forward_onesite(int num, int start, int end, mat & value){
	int i;
	clayer[0]->forward_onesite(num,start,end, value);
	for(i=1;i<layer_num;i++){
		clayer[i]->forward_onesite(clayer[i-1]->update_num, clayer[i-1]->update_start, clayer[i-1]->update_end,clayer[i-1]->update_value_act);
	}
	player->forward_onesite(clayer[layer_num-1]->update_num, clayer[layer_num-1]->update_start, clayer[layer_num-1]->update_end,clayer[layer_num-1]->update_value_act);
}
void conv_set::update(){
	int i;
	for(i=0;i<layer_num;i++){
		clayer[i]->update();
	}
	player->update();
}

void conv_set::back_prop(mat & pre_nodes,mat & delta_pre_nodes, vec & update_Grads){
	int i;
	player->back_prop(clayer[layer_num-1]->delta_node_act);
	for(i=layer_num-1;i>0;i--){ 
		clayer[i]->back_prop(clayer[i-1]->node_act,clayer[i-1]->delta_node_act,update_Grads);
	}
	clayer[0]->back_prop(pre_nodes,delta_pre_nodes,update_Grads);
}

void conv_set::first_back_prop(mat & pre_nodes, vec & update_Grads){
	int i;
	player->back_prop(clayer[layer_num-1]->delta_node_act);
	for(i=layer_num-1;i>0;i--){
		clayer[i]->back_prop(clayer[i-1]->node_act,clayer[i-1]->delta_node_act,update_Grads);
	}
	clayer[0]->first_back_prop(pre_nodes,update_Grads);
}

void conv_set::update_parameter(vec & A){
	int i;
	for(i=0;i<layer_num;i++){
		clayer[i]->update_parameter(A);
	}
}

void conv_set::save_parameter(vec & A){
	int i;
	for(i=0;i<layer_num;i++){
		clayer[i]->update_parameter(A);
	}
}

void conv_set::load_parameter(vec & A){
	int i;
	for(i=0;i<layer_num;i++){
		clayer[i]->update_parameter(A);
	}
}
