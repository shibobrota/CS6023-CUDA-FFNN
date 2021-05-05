
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <vector>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/transform_reduce.h>
#include <thrust/functional.h>


enum LayerType
{
	INPUT,
	DENSE,
	OUTPUT
};

enum ActivationType
{
	SIGMOID,
	RELU,
	TANH,
	SOFTMAX,
	NONE
};

enum OptimizerType 
{
	BATCH_GD, 
	STOCHASTIC_GD, 
	MOMENTUM_BASED_GD
};

enum InitializationType
{
	RANDOM,
	XAVIER,
	ZERO
};

struct Exp
{
	__host__ __device__
		double operator()(const double& x) const {
		return exp(x);
	}
};

struct ReLU {

	/* Unary Operator */
	__host__ __device__
		double operator ()(double data) {
		return max(data, 0.0);
	}
};

struct TanH {
	__host__ __device__
		double operator () (double data) {
		return tanh(data);
	}
};

struct Sigmoid {
	__host__ __device__
		double operator () (double data) {
		return (1.0 / (1 + exp(-data)));
	}
};

struct Softmax {
	double sum;

	Softmax(double _sum) : sum(_sum) {}

	__host__ __device__
		double operator () (double data) {
		return (exp(data) / sum);
	}
};

class Layer {

public:
	
	LayerType type;
	unsigned size;
	ActivationType activation;

	/* Activation Layer Nodes */
	thrust::device_vector<double> d_H;
	thrust::host_vector<double> h_H;

	/* Pre-Activation Layer Nodes */
	thrust::device_vector<double> d_A;

	void applyActivation() {

		switch (activation) {
		case SOFTMAX:
		{
			double sum = thrust::transform_reduce(d_A.begin(), d_A.end(), Exp(), 0.0, thrust::plus<float>());
			thrust::transform(d_A.begin(), d_A.end(), d_H.begin(), Softmax(sum));
			break;
		}
		case RELU:
		{
			thrust::transform(d_A.begin(), d_A.end(), d_H.begin(), ReLU());
			break;
		}
		case TANH:
		{
			thrust::transform(d_A.begin(), d_A.end(), d_H.begin(), TanH());
			break;
		}
		case SIGMOID:
		{
			thrust::transform(d_A.begin(), d_A.end(), d_H.begin(), Sigmoid());
		}
		default:
		{
			printf("Undefined Activation!");
		}
		}
	}

	Layer(LayerType _type, unsigned _size, ActivationType _activation) {
		size = _size;
		activation = _activation;
		type = _type;

		if (type == LayerType::INPUT) {
			h_H = thrust::host_vector<double>(size);
		}
		else {
			d_A = thrust::device_vector<double>(size);
		}
		d_H = thrust::device_vector<double>(size);
	}
};


class Model {
public:
	std::vector<Layer> layers;
	OptimizerType optimizerType;
	InitializationType initializationType;

};


int main()
{
    
	Layer layer(LayerType::DENSE, 20, ActivationType::SOFTMAX);

	thrust::fill(layer.d_A.begin(), layer.d_A.end(), 0.1);

	layer.applyActivation();

	thrust::host_vector<double> temp = layer.d_H;

	thrust::copy(temp.begin(), temp.begin()+20, std::ostream_iterator<double>(std::cout, " "));

    return 0;
}
