#include <iostream>
#include <vector>
#include <cmath>
#include <ctime>
#include <fstream>
#include <random>
#include "time.h"
#include <armadillo>
#include <iomanip>
#include "neural_network.h"
using namespace std;
using namespace arma;

double Entanglement_Entropy(int L){
	double a=0;
	neural_network net_alpha1(1, Num, Kernel);
	net_alpha1.save_parameter();
	neural_network net_alpha2(1, Num, Kernel);
	net_alpha2.load_parameter();
	neural_network net_beta1(1, Num, Kernel);
	net_beta1.load_parameter();
	neural_network net_beta2(1, Num, Kernel);
	net_beta2.load_parameter();
	mat spin(1,N);
	spin.fill(1);
	spin[2]=-1;
	net_alpha1.forward(spin);
	cout<<net_alpha1.output<<endl;
	net_alpha2.forward(spin);
	cout<<net_alpha2.output<<endl;
	net_beta1.forward(spin);
	cout<<net_beta1.output<<endl;
	net_beta2.forward(spin);
	cout<<net_beta2.output<<endl;
	return a;
}

int main(){

	double ee=Entanglement_Entropy(5);
	cout<<ee<<endl;
	return 0;
}