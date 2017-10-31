#include <iostream>
#include <vector>
#include <cmath>
#include <ctime>
#include <fstream>
#include <random>
#include "time.h"
#include <armadillo>
#include <iomanip>
#include "conv_set.h"
#include "fc_layer.h"
#include "final_layer.h"
using namespace std;
using namespace arma;

const double pi=M_PI;
const int N=64;
const int set_num=1; 
vec Num={10};
vec Kernel={5};
class neural_network{
public:
	double h; //transverse field for ising model
	conv_set * clayer[set_num];
	fully_connected_layer * full_layer; //fully_connected layer
	final_layer * flayer;
	double output;
	double update_output;
	mat update_first_value;
	double energy; //energy after monte carlo, for the state;
	int num_par; //number of variational parameters;
	mat Correlation;// correlation function for spin;
	double Mag2;// Magnetization <m^2>
	double corr_length;
	vec Force;
	vec Grads;
	vec update_Grads; // Grads accumulation at each monte carlo sample point
	mat Covariance_matrix; 
	vec Adam_m; //Adam Algorithm 
	vec Adam_v;
	vec NAdam_m; //NAdam Algorithm
	vec NAdam_v;
	vec hat_NAdam_m;
	vec hat_NAdam_v;
	double pbeta1; //product of beta1 for Adam
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


neural_network::neural_network(double hh, vec Num, vec Kernel){
	h=hh;

	int i,in=1, ssize=N;
	int para_start=0;
	num_par=0;
	for(i=0;i<set_num;i++){
		clayer[i]=new conv_set(in,Num[i],ssize,Kernel[i], para_start);
		ssize/=pool_size;
		para_start=clayer[i]->num_end+1;
		num_par+=clayer[i]->num_par;
	}
	para_start=clayer[set_num-1]->num_end+1;
	full_layer=new fully_connected_layer(Num[set_num-1],ssize, full_size, para_start);
	num_par+=full_layer->num_par;
	para_start=full_layer->num_end+1;
	flayer=new final_layer(1,full_layer->size,para_start);// size4=2
	num_par+=flayer->num_par;

	
	cout<<"number of parameters: "<<num_par<<endl;
	if(flayer->num_end+1 != num_par){
		cout<<"number of parameters mistake."<<endl;
		cout<<"num_par: "<<num_par<<" flayer->num_end+1: "<<flayer->num_end+1<<endl;
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


void neural_network::initialize(){
	Grads.fill(0);
	Force.fill(0);
	Covariance_matrix.fill(0);
}

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
// calculate energy of spin model
double neural_network::cal_energy(mat & spin){
	double testenergy=0;
	//double *spin1=new double[N];
	int sweep;
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

//monte carlo for optimization
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

//Update parameters after each monte carlo iteration
double neural_network::Adam_update_parameter(double lambda, double learn_rate, int t, long int T1, long int T2){
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

double neural_network::NAG_update_parameter(double lambda, double learn_rate, long int T1, long int T2){
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

double neural_network::NAdam_update_parameter(double lambda, double learn_rate, int t, long int T1, long int T2){
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


void neural_network::NAdam_learn(){
	int A=500;
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
		lambda=100*pow(0.95,iter);
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
	int A=500;
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
		if(iter>500){
			T1=10000*N;
			T2=10000*N;
		}
		if(iter>600){
			T1=10000*N;
			T2=50000*N;
		}
		preenergy=newenergy;
		double converge_error=Adam_update_parameter(lambda,learn_rate,iter+1,T1,T2);
		cout<<"h is "<<h<<" step is: "<<iter<<endl;
		std::cout << std::fixed;
		cout<<"energy is: "<<std::setprecision(8)<<energy/N<<endl;
		newenergy=energy/N;
		cout<<converge_error<<endl;
		//if(iter>300&&abs(newenergy-preenergy)<0.000001){break;}
	}
	int B=500;
	T1=10000*N;
	T2=10000*N;
	nncount=0;
	int m=50;
	learn_rate=0.01;
	for(t=1;t<B;t++){
		iter=t-1;
		lambda=100*pow(0.95,iter);
		if(lambda<0.00001){
			lambda=0.00001;
		}
		if(iter==200){
			T1*=2;
			T2*=5;
		}
		if(iter>200 &&iter <500){
			
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
		double converge_error=NAG_update_parameter(lambda,learn_rate,T1,T2);
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
			cout<<"precise energy is "<<measure_energy(100000*N,100000*N)<<endl;
		}
		measure++;
		if(measure==200){
			measure=0;
		}

	}
	// We use adam algorithm to update parameters.
	save_parameter();
}

//measurements
double neural_network::measure_energy(long int T1, long int T2){
	int i;
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
    double testenergy=0;
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
    double testenergy=0;
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


