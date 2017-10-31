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
const int pool_size=2;
class pooling_layer{   //average pooling layer
public:
	mat node;
	mat delta_node;
	int num;// number of images,should be equal for in and out; channels
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
	//int sstart=start%pool_kernel;
	//int eend=end%pool_kernel;
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
