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

class conv_layer{
public:
	mat node;
	mat weight;
	vec bias;
	mat node_act;
	mat delta_node_act;
	mat delta_node;
	vec grad_bias;
	mat grad_weight;
	int num_par;
	int innum;
	int outnum;
	int size;// size of image
	int kernel_size; //size for convolutional kernel; we assume it's odd;
	int half_kernel_size; // if kernel_size is 5, half_kernel_size is 2;
	int update_start,update_end,update_num;
	mat update_value;
	mat update_value_act;
	string name;
	int num_start;// We store parameters of the whole neural network as a vector and this is the start index in that vector
	int num_end;
	conv_layer(int x, int y, int z,int kernel, int para_start);
	void  forward(mat & spin);
	void forward_onesite(int num, int start, int end, mat & value);
	void update();
	void back_prop(mat & pre_nodes,mat & delta_pre_nodes, vec & update_Grads); //update gradient and back-propagate to delta, which is the delta value of the former layer.
	void first_back_prop(mat & pre_nodes, vec & update_Grads);// backpropagation for first layer, which doesn't has parent layer;
	void save_parameter(vec & A);
	void load_parameter(vec & A);
	void update_parameter(vec & A);
};

conv_layer::conv_layer(int x,int y,int z,int kernel, int para_start){
	innum=x;
	outnum=y;
	size=z; //size of image
	kernel_size=kernel;
	half_kernel_size=(kernel_size-1)/2;
	node.set_size(outnum,size);
	node_act.set_size(outnum,size);
	weight.set_size(innum*outnum,kernel_size);
	bias.set_size(outnum);
	delta_node.set_size(outnum,size);
	delta_node_act.set_size(outnum,size);
	grad_weight.set_size(innum*outnum,kernel_size);
	grad_bias.set_size(outnum);
	update_value.set_size(outnum,size);
	update_value_act.set_size(outnum,size);
	default_random_engine generator;
  	normal_distribution<double> distribution(0.01,0.001);
  	weight.randu();
  	double a=sqrt(3)/sqrt(size);
  	weight=weight*2*a-a;
  	//weight.fill(0.1);
  	//bias.randn();
  	bias.fill(0);
  	num_start=para_start;
  	num_par=innum*outnum*kernel_size+outnum;
  	num_end=num_start+num_par-1;
	//update_num=updatelen+kernel-1; //updatelen is the number of updated nodes in previous layer
	//update_value.set_size(outnum,update_num);
	//update_value_act.set_size(outnum,update_num);
}

void conv_layer::forward(mat & spin){ // spin are images with innum* size from previous layers
	//Convolution is done with periodic condition;
	int in;
	int out;
	int i;
	int k;
	for(out=0;out<outnum;out++){
		for(i=0;i<size;i++){
			node.at(out,i)=0;
			for(in=0;in<innum;in++){
				for(k=0;k<kernel_size;k++){
					node.at(out,i)+=spin.at(in,ZL((i+k-half_kernel_size),size))*weight.at(in*outnum+out,k);
				}
			}
			node.at(out,i)+=bias[out];
			node_act.at(out,i)=Relu(node.at(out,i));
		}
	}
}

void conv_layer::forward_onesite(int num, int start, int end, mat & preupdate_value){
	const int nn=num+kernel_size-1;
	int out,in;
	int i,j,k;
	int index,preindex;
	update_value_act.fill(0);
	update_value.fill(0);
	if(nn<size){
		update_start=start-half_kernel_size;
		update_end=end+half_kernel_size;
		if(update_start<0){
			update_start+=size;
			update_end+=size;
		}
		update_num=nn;
		
		int eend;
		int sstart;
		for(out=0;out<outnum;out++){
			for(i=0;i<nn;i++){
				index=ZL(i+update_start,size);
				if(i<num){
					eend=i;
				}
				else{
					eend=num-1;
				}
				if(i>=2*half_kernel_size){
					sstart=i-2*half_kernel_size;
				}
				else{
					sstart=0;
				}
				for(in=0;in<innum;in++){
					for(j=sstart;j<=eend;j++){
						preindex=ZL(j+start,size);
						if((j-i+2*half_kernel_size)<0 ||(j-i+2*half_kernel_size)>=kernel_size){cout<<"mistake"<<endl;}
						update_value.at(out,index)+=weight.at(in*outnum+out,j-i+2*half_kernel_size)*preupdate_value.at(in,preindex);
					}
				}
				update_value_act.at(out,index)=Relu(update_value.at(out,index)+node.at(out,index))-node_act.at(out,index);
			}
		}
	}
	else{
		update_num=size;
		update_start=0;
		update_end=size-1;
		update_value.fill(0);

		for(out=0;out<outnum;out++){
			for(i=0;i<size;i++){
				for(in=0;in<innum;in++){
					for(k=0;k<kernel_size;k++){
						update_value.at(out,i)+=preupdate_value.at(in,ZL((i+k-half_kernel_size),size))*weight.at(in*outnum+out,k);
					}
				}
				update_value_act.at(out,i)=Relu(update_value.at(out,i)+node.at(out,i))-node_act.at(out,i);
			}
		}
	}	
}

void conv_layer::update(){ //update forward_one_site
	int i;
	int out;
	int index;
	for(out=0;out<outnum;out++){
		for(i=0;i<update_num;i++){
			index=ZL((i+update_start),size);
			node.at(out,index)+=update_value.at(out,index);
			node_act.at(out,index)+=update_value_act.at(out,index);
			//node_act[(i+update_start)&PBC+size*out]=tanh(node[(i+update_start)&PBC+size*out]);
		}
	}
}

void conv_layer::load_parameter(vec & A){
	int i;
	for(i=0;i<innum*outnum*kernel_size;i++){
		weight[i]=A[i+num_start];
	}
	for(i=0;i<outnum;i++){
		bias[i]=A[innum*outnum*kernel_size+i+num_start];
	}
}

void conv_layer::update_parameter(vec & A){
	int i;
	for(i=0;i<innum*outnum*kernel_size;i++){
		weight[i]+=A[i+num_start];
	}
	for(i=0;i<outnum;i++){
		bias[i]+=A[innum*outnum*kernel_size+i+num_start];
	}
}
void conv_layer::save_parameter(vec & A){
	int i;
	for(i=0;i<innum*outnum*kernel_size;i++){
		A[i+num_start]=weight[i];
	}
	for(i=0;i<outnum;i++){
		A[innum*outnum*kernel_size+i+num_start]=bias[i];
	}
}

void conv_layer::back_prop(mat & pre_nodes, mat & delta_pre_nodes, vec & update_Grads){
	//Convolution is done with periodic condition;
	int in;
	int out;
	int i;
	int k;
	for(out=0;out<outnum;out++){
		for(i=0;i<size;i++){
			delta_node.at(out,i)=grad_Relu(node.at(out,i))*delta_node_act.at(out,i);
		}
	}
	grad_weight.fill(0);
	delta_pre_nodes.fill(0);
	grad_bias.fill(0);
	for(out=0;out<outnum;out++){
		for(i=0;i<size;i++){
			for(in=0;in<innum;in++){
				for(k=0;k<kernel_size;k++){
					delta_pre_nodes.at(in,ZL((i+k-half_kernel_size),size))+=weight.at((in*outnum+out),k)*delta_node.at(out,i);
					grad_weight.at(in*outnum+out,k)+=pre_nodes.at(in,ZL((i+k-half_kernel_size),size))*delta_node.at(out,i);
				}
			}
			grad_bias[out]+=delta_node.at(out,i);
		}
	}

	for(i=0;i<innum*outnum*kernel_size;i++){
		update_Grads[i+num_start]=grad_weight[i];
	}
	for(i=0;i<outnum;i++){
		update_Grads[innum*outnum*kernel_size+i+num_start]=grad_bias[i];
	}

}

void conv_layer::first_back_prop(mat &pre_nodes, vec & update_Grads){
	//Convolution is done with periodic condition;
	int in;
	int out;
	int i;
	int k;
	for(out=0;out<outnum;out++){
		for(i=0;i<size;i++){
			delta_node.at(out,i)=grad_Relu(node.at(out,i))*delta_node_act.at(out,i);
		}
	}
	grad_weight.fill(0);
	grad_bias.fill(0);
	for(out=0;out<outnum;out++){
		for(i=0;i<size;i++){
			for(in=0;in<innum;in++){
				for(k=0;k<kernel_size;k++){
					grad_weight.at(in*outnum+out,k)+=pre_nodes.at(in,ZL((i+k-half_kernel_size),size))*delta_node.at(out,i);
				}
			}
			grad_bias[out]+=delta_node.at(out,i);
		}
	}

	for(i=0;i<innum*outnum*kernel_size;i++){
		update_Grads[i+num_start]=grad_weight[i];
	}
	for(i=0;i<outnum;i++){
		update_Grads[innum*outnum*kernel_size+i+num_start]=grad_bias[i];
	}
}





