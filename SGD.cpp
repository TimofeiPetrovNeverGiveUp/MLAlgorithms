#include <cmath>
#include <vector>
#include <random>
#include <iostream>
#include <chrono>

//sigmoid loss function
double loss(std::vector<double> &w, std::vector<double> &x, double y)
{
	double M = 0.0;
	for (int i = 0; i < 3; ++i)
	{
		M += w[i] * x[i];
	}
	M *= y;
	double result = 2 / (1 + std::exp(M));
	return result;
}

//d(loss)
std::vector<double> dloss(std::vector<double> &w, std::vector<double> &x, double y)
{
	double M = 0.0;
	for (int i = 0; i < 3; ++i)
	{
		M += w[i] * x[i];
	}
	M *= y;
	double result = pow((1 + std::exp(M)), -2);
	result *= -2;
	result *= std::exp(M);
	std::vector<double> x_copy = { x[0],x[1],x[2] };
	for (int i = 0; i < 3; ++i)
	{
		x_copy[i] *= result * y;
	}
	return x_copy;
}

int main()
{
	//time start
	auto start = std::chrono::high_resolution_clock::now();

	//for random
	std::random_device rd;
	std::mt19937 gen(rd());

	//training sample
	std::vector<std::vector<double>> x_train =
	{ {10,50,1},
	{20,30,1},
	{25,30,1},
	{20,60,1},
	{15,70,1},
	{40,40,1},
	{30,45,1},
	{20,45,1},
	{40,30,1},
	{7,35,1} };
	std::vector<double> y_train = { -1,1,1,-1,-1,1,1,-1,1,-1 };

	int n_train = x_train.size(); //size of training sample
	std::uniform_int_distribution<int> dist(0, n_train - 1);//creating a distribution
	std::vector<double> w = { 0.0, 0.0, 0.0 }; //start ratio
	double nt = 0.0005; //convergence step SGD
	double lm = 0.01; //lambda for Q
	int N = 1000; //iteration count of SGD

	double Q = 0.0; //quality
	for (int i = 0; i < n_train; ++i)
	{
		Q += loss(w, x_train[i], y_train[i]);
	}
	Q /= n_train;
	
	for (int i = 0; i < N; ++i)
	{
		int k = dist(gen); //random index
		double ek = loss(w, x_train[k], y_train[k]); //loss
		std::vector<double> wf = dloss(w, x_train[k], y_train[k]); //SGD
		for (int i = 0; i < 3; ++i)
		{
			w[i] = w[i] - nt*wf[i];
		}
		Q = lm * ek + (1 - lm) * Q; //quality
	}
	//time stop
	auto end = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
	std::cout << "Time: " << duration.count() / 1000.0 << " seconds" << std::endl;
	for (int i = 0; i < 3; ++i)
	{
		std::cout << w[i] << " ";
	}
	std::cout << "\n" << "Q: " << Q;
	return 0;
}

