#include <iostream>
#include <vector>
#include <cmath>
#include <ctime>
#include <fstream>
#include <random>
#include "time.h"
#include <armadillo>
#include <iomanip>
using namespace std;
using namespace arma;



const int N=64; //must be 2^x form
const int PBC=N-1;
const double pi=M_PI;
const int pool_size=2;
const int full_size=5;
const int layer_num=1; // number of layers in conv_set
const int set_num=3; //number of conv_sets in neural_net;
vec Num={20,20,20};
vec Kernel={9,7,5};

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
  	int i;
  	weight.randn();
  	weight=weight/10;
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

class pooling_layer{   //average pooling layer
public:
	mat node;
	mat delta_node;
	int num;// number of images,should be equal for in and out
	int size; //size of image
	int update_start;
	int update_end;
	mat update_value;
	string name;
	int update_num;
	int pool_kernel;
	pooling_layer(int n,  int pre_size,  int poolsize);
	void forward(mat & pre_nodes,int pre_size);
	void forward_onesite(int nn, int start, int end, mat & value);
	void update();
	void back_prop(mat & delta_pre_nodes);
};




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
	weight.randn();
	weight=weight/100;
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

class neural_network{
public:
	double h; //transverse field for ising model
	conv_set * clayer[set_num];
	fully_connected_layer * full_layer; //fully_connected layer
	final_layer * flayer;
	//double * spin_input;
	double output;
	double update_output;
	mat update_first_value;
	double energy; //energy after monte carlo, for the state;
	int num_par; //number of variational parameters;
	//double * Force; //<EO_k>
	//double * Grads; //<O_k>
	//double *Covariance_matrix;
	mat Correlation;// correlation function for spin;
	double Mag2;// Magnetization <m^2>
	double corr_length;
	vec Force;
	vec Grads;
	vec update_Grads; // Grads accumulation at each monte carlo sample point
	mat Covariance_matrix; 
	vec Adam_m; //Adam Algorithm 
	vec Adam_v;
	vec NAdam_m;
	vec NAdam_v;
	vec hat_NAdam_m;
	vec hat_NAdam_v;
	double pbeta1; //product of beta1
	double pbeta2;
	double beta1t; //beta1 at time t
	double beta2t;
	vec update_x;
	vec hat_update_x;
	vec update_xpre; // For NAG algorithm, update in the previous step, initialized as zero.
	double local_energy; //energy for the current spin configuration;
	neural_network(double hh, vec Num, vec Kernel);
	void initialize();// initialize Force, Grads, Covariance_matrix for a new monte_carlo iteration.
	void forward(mat & spin);
	double forward_onesite(int x, double y);
	void update();
	void monte_carlo(long int T1, long int T2);
	double cal_energy(mat & spin);
	void back_prop(mat & spin);// backpropagation to update gradient
	void back_prop_accumu();//accumulate gradients to force and covariance_matrix
	void update_parameter(int index); //update_parameter by update_x
	double Adam_update_parameter(double lambda,double learn_rate, int t,long int T1, long int T2);//update parameters after each monte carlo, Adam algorithm
	double NAG_update_parameter(double lambda, double learn_rate, long int T1, long int T2);// update parameters using NAG algorithm
	double NAdam_update_parameter(double lambda, double learn_rate, int t, long int T1, long int T2);
	void learn();
	void NAdam_learn();
	void save_parameter();
	void load_parameter();
	void measure_mc(long int T1, long int T2);
	double measure_energy(long int T1,long int T2);
};

pooling_layer::pooling_layer(int n,  int pre_size, int poolsize){
	num=n;
	pool_kernel=poolsize;
	size=pre_size/pool_kernel;
	node.set_size(num,size);
	delta_node.set_size(num,size);
	update_value.set_size(num,size);
	//update_value.set_size(num,updatelen);
}

void pooling_layer::forward(mat & pre_nodes, int pre_size){
	int i,j,k;
	for(i=0;i<num;i++){
		for(j=0;j<size;j++){
			node.at(i,j)=0;
			for(k=0;k<pool_kernel;k++){
				node.at(i,j)+=pre_nodes.at(i,j*pool_kernel+k)/pool_kernel;
			}
		}
	}
}

void pooling_layer::forward_onesite(int nn, int start, int end, mat & preupdate_value){
	update_start=start/pool_kernel;
	update_end=end/pool_kernel;
	update_num=update_end-update_start+1;
	int sstart=start%pool_kernel;
	int eend=end%pool_kernel;
	int i,j,k;
	int index,preindex;
	update_value.fill(0);
	for(i=0;i<num;i++){
		for(j=0;j<update_num;j++){
			index=ZL(j+update_start,size);
			for(k=0;k<pool_kernel;k++){
				preindex=index*pool_kernel+k;
					//cout<<"index is "<<index<<" value is "<<value.at(i,index)<<endl;
				update_value.at(i,index)+=preupdate_value.at(i,preindex)/pool_kernel;
			}
		}
	}
}

void pooling_layer::update(){
	int i,j;
	int index;
	for(i=0;i<num;i++){
		for(j=0;j<update_num;j++){
			index=ZL(j+update_start,size);
			node.at(i,index)+=update_value.at(i,index);
		}
	}
}

void pooling_layer::back_prop(mat &delta_pre_nodes){
	int i,j,k;
	for(i=0;i<num;i++){
		for(j=0;j<size;j++){
			for(k=0;k<pool_kernel;k++){
				delta_pre_nodes.at(i,j*pool_kernel+k)=delta_node.at(i,j)/pool_kernel;
			}
		}
	}
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
		int index;
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

void conv_layer::update(){
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
final_layer::final_layer(int x,int y, int para_start){
	innum=x;
	size=y;
	weight.set_size(innum,size);
	grad_weight.set_size(innum,size);
	bias=0;
	weight.randn();
	weight=weight/100;
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

void final_layer::update(){
	output+=update_output;
}

neural_network::neural_network(double hh, vec Num, vec Kernel){
	h=hh;

	int i,in=1, ssize=N,j;
	int para_start=0;
	num_par=0;
	for(i=0;i<set_num;i++){
		clayer[i]=new conv_set(in,Num[i],ssize,Kernel[i], para_start);
		ssize/=pool_size;
		para_start=clayer[i]->num_end;
		num_par+=clayer[i]->num_par;
	}

	para_start=clayer[set_num-1]->num_end+1;
	full_layer=new fully_connected_layer(Num[set_num-1],ssize, full_size, para_start);
	num_par+=full_layer->num_par;
	para_start=full_layer->num_end+1;
	flayer=new final_layer(1,full_layer->size,para_start);// size4=2
	num_par+=flayer->num_par;

	

	if(flayer->num_end+1 != num_par){
		cout<<"number of parameters mistake."<<endl;
		cout<<"num_par is "<<num_par<<" flayer->num_end+1 "<<flayer->num_end+1<<endl;
	}
	Force.zeros(num_par);
	Grads.zeros(num_par);//Gradient of parameters
	update_Grads.zeros(num_par);//used at each monte_carlo configuration
	Covariance_matrix.zeros(num_par,num_par);
	Adam_m.zeros(num_par);//Adam algorithm
	Adam_v.zeros(num_par);//Adam algorithm
	NAdam_m.zeros(num_par);//NAdam algorithm
	NAdam_v.zeros(num_par);//NAdam algorithm
	hat_NAdam_m.zeros(num_par);
	hat_NAdam_v.zeros(num_par);
	pbeta1=1;
	pbeta2=1;
	update_x.zeros(num_par);
	hat_update_x.zeros(num_par);
	update_xpre.zeros(num_par);
	Correlation.set_size(N,N);
	Mag2=0;
	corr_length=0;
	update_first_value.set_size(1,N);
	update_first_value.fill(0);
	//spin_input=new double(N);

	ofstream fout;
	fout.open("network.txt");
	fout<<"number of parameters: "<<num_par<<endl;
	
	fout<<flush;
	fout.close();


}

void neural_network::forward(mat & spin){


	int i;
	clayer[0]->forward(spin);
	for(i=1;i<set_num;i++){
		clayer[i]->forward(clayer[i-1]->player->node);
	}


	full_layer->forward(clayer[set_num-1]->player->node);
	flayer->forward(full_layer->node);
	
	output=softplus(flayer->output);
}

double neural_network::forward_onesite(int x, double y){
	int start=x;
	int end=x;
	update_first_value[x]=y;
	int num=1;
	int i;
	clayer[0]->forward_onesite(num,start,end, update_first_value);

	for(i=1;i<set_num;i++){
		clayer[i]->forward_onesite(clayer[i-1]->player->update_num,clayer[i-1]->player->update_start,clayer[i-1]->player->update_end,clayer[i-1]->player->update_value);
	}
	


	full_layer->forward_onesite(clayer[set_num-1]->player->update_num,clayer[set_num-1]->player->update_start,clayer[set_num-1]->player->update_end,clayer[set_num-1]->player->update_value);
	flayer->forward_onesite(full_layer->update_num,full_layer->update_start,full_layer->update_end,full_layer->update_value);
	update_output=softplus(flayer->output+flayer->update_output);

	update_first_value[x]=0;


	
	return update_output;
}
void neural_network::update(){
	int i;
	for(i=0;i<set_num;i++){
		clayer[i]->update();
	}
	full_layer->update();
	flayer->update();
	output=update_output;


}
double neural_network::cal_energy(mat & spin){
	double testenergy=0;
	//double *spin1=new double[N];
	int i,sweep;
	double out1;
	
    for(sweep=0;sweep<N;sweep++){
      testenergy+=-spin[sweep]*spin[(sweep+1)%N]*output;
      forward_onesite(sweep,-spin[sweep]*2);
      out1=update_output;
      testenergy-=h*out1;
     }
    testenergy/=output;
    local_energy=testenergy;
    return testenergy;
}


void neural_network::initialize(){
	int i,j;
	Grads.fill(0);
	Force.fill(0);
	Covariance_matrix.fill(0);
}
//Back propagation code
void neural_network::back_prop_accumu(){
	Grads+=update_Grads;
	Force+=update_Grads*local_energy;
	Covariance_matrix+=update_Grads*update_Grads.t();
}
void neural_network::back_prop(mat & spin){
	//cout<<"success 0"<<endl;
	//flayer->delta_output=grad_sigmoid(flayer->output)/output;  //\frac{\delta \psi}{\psi}
	flayer->delta_output=grad_softplus(flayer->output);  //\frac{\delta \psi}{\psi}
	//cout<<"success 0.1"<<endl;
	//flayer->delta_output=pow(1./cosh(flayer->output),2)/output;
	flayer->back_prop(full_layer->node,full_layer->delta_node,update_Grads);
	full_layer->back_prop(clayer[set_num-1]->player->node,clayer[set_num-1]->player->delta_node, update_Grads);

	int i;
	for(i=set_num-1;i>0;i--){
		clayer[i]->back_prop(clayer[i-1]->player->node,clayer[i-1]->player->delta_node,update_Grads);
	}
	clayer[0]->first_back_prop(spin, update_Grads);
}


double neural_network::NAdam_update_parameter(double lambda, double learn_rate, int t, long int T1, long int T2){
	int i,j;
	if(t==1){
		beta1t=0.99;
		beta2t=0.999;
	}
	if(t>1){
		beta1t=0.99*(1-0.5*pow(0.96,double(t)/250));
	}
	pbeta1*=beta1t;
	pbeta2*=0.999;
	monte_carlo(T1,T2);
	mat A;
	A.eye(num_par,num_par);
	
	Force=Force-energy*Grads;
	Covariance_matrix-=Grads*Grads.t();
	Covariance_matrix+=lambda*A;
	update_x=solve(Covariance_matrix,Force);
	
	double  epsilon=0.00000001,beta1=0.99,beta2=0.999, nextbeta1;
	nextbeta1=0.99*(1-0.5*pow(0.96,double(t+1)/250));
	hat_update_x=update_x/(1-pbeta1);
	NAdam_m*=beta1;
	NAdam_m+=(1-beta1)*update_x;
	hat_NAdam_m=NAdam_m/(1-pbeta1*nextbeta1);
	NAdam_v*=beta2;
	NAdam_v+=(1-beta2)*(update_x%update_x);
	hat_NAdam_v=NAdam_v/(1-pbeta2);

	update_x=-learn_rate*((1-beta1t)*hat_update_x+nextbeta1*hat_NAdam_m)/(sqrt(hat_NAdam_v)+epsilon);

	update_parameter(0);
	return max(abs(update_x));
}


double neural_network::Adam_update_parameter(double lambda, double learn_rate, int t, long int T1, long int T2){
	int i,j;
	
	monte_carlo(T1,T2);
	mat A;
	A.eye(num_par,num_par);
	
	Force=Force-energy*Grads;
	Covariance_matrix-=Grads*Grads.t();
	Covariance_matrix+=lambda*A;
	update_x=solve(Covariance_matrix,Force);
	
	double beta1=0.9, beta2=0.999, epsilon=0.00000001, alpha=learn_rate,alpha_t;
	Adam_m*=beta1;
	Adam_m+=(1-beta1)*update_x;
	Adam_v*=beta2;
	Adam_v+=(1-beta2)*(update_x%update_x);
	alpha_t=alpha*sqrt(1-pow(beta2,t))/(1-pow(beta1,t));
	update_x=-alpha_t*Adam_m/(sqrt(Adam_v)+epsilon*sqrt(1-pow(beta2,t)));

	update_parameter(0);
	return max(abs(update_x));
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
void neural_network::load_parameter(){
	int i;
	vec A(num_par);
	stringstream stream;
	stream << fixed << setprecision(3) << h;
	string s = stream.str();
	string name="parameter_h=";
	name=name+s+".mat";
	A.load(name,arma_ascii);
	for(i=0;i<set_num;i++){
		clayer[i]->load_parameter(A);
	}
	full_layer->load_parameter(A);
	flayer->load_parameter(A);
}
void neural_network::save_parameter(){
	int i;
	vec A(num_par);
	for(i=0;i<set_num;i++){
		clayer[i]->save_parameter(A);
	}
	full_layer->save_parameter(A);
	flayer->save_parameter(A);
	stringstream stream;
	stream << fixed << setprecision(3) << h;
	string s = stream.str();
	string name="parameter_h=";
	name=name+s+".mat";
	A.save(name,arma_ascii);
}
void neural_network::update_parameter(int index){
	int i;
	if(index==0){
		for(i=0;i<set_num;i++){
			clayer[i]->update_parameter(update_x);
		}
		full_layer->update_parameter(update_x);
		flayer->update_parameter(update_x);
	}
	if(index==1){
		for(i=0;i<set_num;i++){
			clayer[i]->update_parameter(update_xpre);
		}
		full_layer->update_parameter(update_xpre);
		flayer->update_parameter(update_xpre);
	}
}

double neural_network::NAG_update_parameter(double lambda, double learn_rate, long int T1, long int T2){
	int i,j;
	double gamma=0.9;
	update_xpre=gamma*update_x;
	update_parameter(1);
	monte_carlo(T1,T2);
	mat A;
	A.eye(num_par,num_par);
	
	Force=Force-energy*Grads;
	//Grads.print();
	//Covariance_matrix.print();
	Covariance_matrix-=Grads*Grads.t();
	Covariance_matrix+=lambda*A;
	//Grads.print();
	update_x=-learn_rate*solve(Covariance_matrix,Force);
	update_x+=update_xpre;
	update_parameter(0);
	return max(abs(update_x));
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
					double a=pre_nodes.at(in,ZL((i+k-half_kernel_size),size))*delta_node.at(out,i);
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
void neural_network::NAdam_learn(){
	int A=1000;
	int period=0;
	double rate0=1;
	int t;
	int iter;
	double lambda;
	double learn_rate=0.01;
	long int T1;
	long int T2;
	T1=100*N;
	T2=100*N;
	double preenergy=0;
	double newenergy=0;
	int mcount=0;
	int nncount=0;
	int measure=0;
	//double learn_rate=0.5;
	lambda=90;
	for(iter=0;iter<A;iter++){
		
		if(iter>50){
			learn_rate=0.001;
		}
		if(iter>200){
			learn_rate=0.0001;
		}
		lambda*=0.96;
		if(lambda<0.001){
			lambda=0.001;
		}
		if(iter==350){
			T1*=2;
			T2*=4;
		}
		preenergy=newenergy;
		double converge_error=NAdam_update_parameter(lambda,learn_rate,iter+1,T1,T2);
		cout<<"h is "<<h<<" step is: "<<iter<<endl;
		std::cout << std::fixed;
		cout<<"energy is: "<<std::setprecision(8)<<energy/N<<endl;
		newenergy=energy/N;
		cout<<converge_error<<endl;
		if(iter>300&&abs(newenergy-preenergy)<0.000001){break;}
	}
	int B=500;
	T1=10000*N;
	T2=10000*N;
	nncount=0;
	int m=50;
	learn_rate=0.001;
	for(t=1;t<B;t++){
		iter=t-1;
		lambda=100*pow(1,iter);
		if(lambda<0.00001){
			lambda=0.00001;
		}
		if(iter==200){
			T1*=5;
			T2*=5;
		}
		if(iter>200 &&iter <=300){
			if(mcount==0){
				T1*=2;
				T2*=2;
			}
			mcount++;
			if(mcount==m){
				mcount=0;
			}
		}
		if(iter>350 &&iter <600){
			if(mcount==0){
				T1*=1;
				T2*=2;
			}
			mcount++;
			if(mcount==100){
				mcount=0;
			}
			T1=20000*N;
			T2=200000*N;
		}
		
		if(iter>100){
			learn_rate=0.001;
		}
		
		if(iter>400){
			learn_rate=0.0001;
		}
		preenergy=newenergy;
		double converge_error=Adam_update_parameter(lambda,learn_rate,t,T1,T2);
		cout<<"h is "<<h<<" step is: "<<iter<<endl;
		std::cout << std::fixed;
		newenergy=energy/N;
		cout<<"energy is: "<<std::setprecision(8)<<energy/N<<endl;
		cout<<converge_error<<endl;
		if(iter>300&&abs(newenergy-preenergy)<0.000001){
			nncount++;
			cout<<"nncount is "<<nncount<<endl;
			cout<<"T1 is "<<T1/N<<"T2 is "<<T2/N<<endl;
		}
		if(measure==0 && iter>300){
		//	cout<<"precise energy is "<<measure_energy(100000*N,500000*N)<<endl;
		}
		measure++;
		if(measure==100){
			measure=0;
		}

	}
	// We use adam algorithm to update parameters.
	save_parameter();
}
void neural_network::learn(){
	int A=450;
	int period=0;
	double rate0=1;
	int t;
	int iter;
	double lambda;
	double learn_rate=0.5;
	long int T1;
	long int T2;
	T1=100*N;
	T2=100*N;
	double preenergy=0;
	double newenergy=0;
	int mcount=0;
	int nncount=0;
	int measure=0;
	//double learn_rate=0.5;
	lambda=90;
	for(iter=0;iter<A;iter++){
		if(iter>20){
			learn_rate=0.1;
		}
		if(iter>50){
			learn_rate=0.05;
		}
		if(iter>100){
			learn_rate=0.001;
		}
		if(iter>150){
			learn_rate=0.0001;
		}
		if(iter>200){
			learn_rate=0.0001;
		}
		if(iter>400){
			learn_rate=0.00001;
		}
		lambda*=0.96;
		if(lambda<0.001){
			lambda=0.001;
		}
		if(iter==350){
			T1*=2;
			T2*=4;
		}
		preenergy=newenergy;
		double converge_error=NAG_update_parameter(lambda,learn_rate,T1,T2);
		cout<<"h is "<<h<<" step is: "<<iter<<endl;
		std::cout << std::fixed;
		cout<<"energy is: "<<std::setprecision(8)<<energy/N<<endl;
		newenergy=energy/N;
		cout<<converge_error<<endl;
		if(iter>300&&abs(newenergy-preenergy)<0.000001){break;}
	}
	int B=2000;
	T1=10000*N;
	T2=10000*N;
	nncount=0;
	int m=50;
	learn_rate=0.05;
	for(t=1;t<B;t++){
		iter=t-1;
		lambda=100*pow(0.94,iter);
		if(lambda<0.00001){
			lambda=0.00001;
		}
		if(iter==300){
			T1*=10;
			T2*=50;
		}
		if(iter>200 &&iter <1000){
			
			if(mcount==0){
				T1*=1;
				T2*=2;
			}
			mcount++;
			if(mcount==m){
				mcount=0;
			}
		}
		if(iter>500){
			T1=100000*N;
			T2=1000000*N;
		}
		if(iter>1000){
			T1=100000*N;
			T2=1000000*N;
		}
		

		if(iter>20){
			learn_rate=0.01;
		}
		if(iter>50){
			learn_rate=0.005;
		}
		if(iter>100){
			learn_rate=0.001;
		}
		if(iter>150){
			learn_rate=0.0001;
		}
		if(iter>200){
			learn_rate=0.00001;
		}
		if(iter>400){
			learn_rate=0.000001;
		}
		preenergy=newenergy;
		double learn_rate=0.001;
		double converge_error=Adam_update_parameter(lambda,learn_rate,t,T1,T2);
		cout<<"h is "<<h<<" step is: "<<iter<<endl;
		std::cout << std::fixed;
		newenergy=energy/N;
		cout<<"energy is: "<<std::setprecision(8)<<energy/N<<endl;
		cout<<converge_error<<endl;
		if(iter>300&&abs(newenergy-preenergy)<0.000001){
			nncount++;
			cout<<"nncount is "<<nncount<<endl;
			cout<<"T1 is "<<T1/N<<"T2 is "<<T2/N<<endl;
		}
		if(measure==0){
			cout<<"precise energy is "<<measure_energy(100000*N,1000000*N)<<endl;
		}
		measure++;
		if(measure==100){
			measure=0;
		}

	}
	// We use adam algorithm to update parameters.
	save_parameter();
}

void neural_network::monte_carlo(long int T1, long int T2){
	int i;
	mat spin(1,N);
	spin.fill(1);
	int corr=0;
	initialize();
	std::mt19937 rng;
    rng.seed(std::random_device()());
	auto real_rand = std::bind(std::uniform_real_distribution<double>(0,1),mt19937(rng));
	auto dice_rand = std::bind(std::uniform_int_distribution<int>(0,N-1),mt19937(rng));
    double total_energy=0;
    double testenergy;
    forward(spin);
    double f1=output;
    int counter=0;
    int step;
    int error_correction=0;
    for(step=0;step<T1+T2;step++){
      i=dice_rand();
      double f2=forward_onesite(i,-2*spin[i]);
      double pp=(f2/f1)*(f2/f1);
      double test=real_rand();
      if(test<pp){
        spin[i]*=-1;
        f1=f2;
        update();
        if(abs(f1-output)>0.00001){cout<<"update mistake!";}
        error_correction++;
        if(error_correction==100000){
        	forward(spin);
        }
  	  }
      if(step==T1){
        testenergy=cal_energy(spin);
        forward(spin);
        if(abs(f1-output)>0.000001){
        	cout<<"update mistake2!";
        	cout<<abs(f1-output)<<endl;
        }
      
      }
      if(step>T1){
      	if(corr==0){
      		testenergy=cal_energy(spin);
	        total_energy+=testenergy;
	        back_prop(spin);
	        back_prop_accumu();
	        counter+=1;
        }
        corr++;
        if(corr==150){
        	corr=0;
        }
      }
    }
   	total_energy/=counter;
    energy=total_energy;
    Grads=Grads/counter;
	Force/=counter;
	Covariance_matrix/=counter;
}

double neural_network::measure_energy(long int T1, long int T2){
	int i,k;
	int accept=0;
	mat spin(1,N);
	spin.fill(1);
	int corr=0;
	initialize();
	std::mt19937 rng;
    rng.seed(std::random_device()());
	auto real_rand = std::bind(std::uniform_real_distribution<double>(0,1),mt19937(rng));
	auto dice_rand = std::bind(std::uniform_int_distribution<int>(0,N-1),mt19937(rng));
    double total_energy=0;
    double testenergy;
    forward(spin);
    double f1=output;
    int counter=0;
    int step;
    int error_correction=0;
    for(step=0;step<T1+T2;step++){
      i=dice_rand();
      double f2=forward_onesite(i,-2*spin[i]);
      double pp=(f2/f1)*(f2/f1);
      double test=real_rand();
      if(test<pp){
      	accept++;
        spin[i]*=-1;
        f1=f2;
        update();
        if(abs(f1-output)>0.00001){cout<<"update mistake!";}
        if(step>T1){
          testenergy=cal_energy(spin);
        }
        error_correction++;
        if(error_correction==10000){
        	forward(spin);
        }
  	  }
      if(step==T1){
        testenergy=cal_energy(spin);
        forward(spin);
        if(abs(f1-output)>0.000001){
        	cout<<"update mistake2!";
        	cout<<abs(f1-output);
        }
      
      }
      if(step>T1){
      	if(corr==0){
	        total_energy+=testenergy;
	        counter+=1;
        }
        corr++;
        if(corr==130){
        	corr=0;
        }
      }
    }
   	total_energy/=counter;
    cout<<"accept ratio is "<<double(accept)/(T1+T2)<<endl;
    return total_energy/N;
}




void neural_network::measure_mc(long int T1, long int T2){
	int i,k;
	int accept=0;
	mat spin(1,N);
	mat testcorr;
	testcorr.set_size(N,N);
	spin.fill(1);
	testcorr.fill(1);
	int corr=0;
	initialize();
	std::mt19937 rng;
    rng.seed(std::random_device()());
	auto real_rand = std::bind(std::uniform_real_distribution<double>(0,1),mt19937(rng));
	auto dice_rand = std::bind(std::uniform_int_distribution<int>(0,N-1),mt19937(rng));
    double total_energy=0;
    double testenergy;
    forward(spin);
    double f1=output;
    int counter=0;
    int step;
    int error_correction=0;
    for(step=0;step<T1+T2;step++){
      i=dice_rand();
      double f2=forward_onesite(i,-2*spin[i]);
      double pp=(f2/f1)*(f2/f1);
      double test=real_rand();
      if(test<pp){
      	accept++;
        spin[i]*=-1;
        f1=f2;
        update();
      	for(k=0;k<N;k++){
      		testcorr.at(i,k)*=-1;
      		testcorr.at(k,i)*=-1;
      	}
        if(abs(f1-output)>0.00001){cout<<"update mistake!";}
        if(step>T1){
          testenergy=cal_energy(spin);
        }
        error_correction++;
        if(error_correction==10000){
        	forward(spin);
        }
  	  }
      if(step==T1){
        testenergy=cal_energy(spin);
        forward(spin);
        if(abs(f1-output)>0.000001){
        	cout<<"update mistake2!";
        	cout<<abs(f1-output);
        }
      
      }
      if(step>T1){
      	if(corr==0){
	        total_energy+=testenergy;
	        counter+=1;
	        Correlation+=testcorr;
        }
        corr++;
        if(corr==130){
        	corr=0;
        }
      }
    }
    cout<<"suc5"<<endl;
    Correlation/=counter;
   	total_energy/=counter;
    energy=total_energy;
    Mag2=0;
    corr_length=0;
    double S0=0,S1=0;
    for(i=0;i<N;i++){
    	for(k=0;k<N;k++){
    		S0+=Correlation.at(i,k);
    		S1+=Correlation.at(i,k)*cos(2*pi*abs(i-k)/N);
    	}
    }
    corr_length=N/(2*pi)*sqrt(S0/S1-1);
    Mag2=S0/(N*N);
    cout<<"accept ratio is "<<double(accept)/(T1+T2)<<endl;
}















int main (){
	clock_t time;
	mat spin(1,N);
	int i;
	time= clock();
	spin.fill(1);
	neural_network net(1, Num, Kernel);
	//net.load_parameter();
	spin[5]=-1;
	net.forward(spin);
	cout<<"output is";
	cout<<setprecision(12)<<net.output<<endl;
	
	spin[7]=-1;
	
	
	cout<<setprecision(15)<<net.forward_onesite(7,-2)<<endl;
	net.update();
	cout<<setprecision(15)<<net.output<<endl;
	net.forward(spin);
	cout<<setprecision(15)<<net.output<<endl;

	//cout<<"energy "<<net.measure_energy(10000*N,10000*N)<<endl;
	net.NAdam_learn();

	cout<<"success"<<endl;
	time=clock()-time;
	time=(double(time))/CLOCKS_PER_SEC;
	cout<<"time needed is "<<time<<"s"<<endl;
	cout<<CLOCKS_PER_SEC<<endl;
	return 0;
}