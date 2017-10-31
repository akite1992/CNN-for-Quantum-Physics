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



int main (){
	clock_t time;
	mat spin(1,N);
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
	//net.learn();

	cout<<"success"<<endl;
	time=clock()-time;
	time=(double(time))/CLOCKS_PER_SEC;
	cout<<"time needed is "<<time<<"s"<<endl;
	cout<<CLOCKS_PER_SEC<<endl;
	return 0;
}
