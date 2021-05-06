
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <vector>
#include <string>
#include<time.h>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/transform_reduce.h>
#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/random.h>



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

struct prg
{
	double a, b;
	thrust::default_random_engine rng;
	thrust::uniform_real_distribution<double> dist;

	__host__ __device__
	prg(unsigned seed, double _a = 0.0, double _b = 1.0) : a(_a), b(_b) {
		this->rng = thrust::default_random_engine(seed);
		this->dist = thrust::uniform_real_distribution<double>(a, b);
	};

	__host__ __device__
	double operator()(unsigned d)
	{
		this->rng.discard(d);
		return this->dist(rng);
	}
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

	bool isValid() {
		if (layers.at(0).type != LayerType::INPUT || layers.at(layers.size() - 1).type != LayerType::OUTPUT) {
			return false;
		}
		return true;
	}

public:
	
	OptimizerType optimizerType;
	InitializationType initializationType;
	double learningRate;
	unsigned epochs;
	unsigned batchSize;
	std::vector<thrust::host_vector<double>> h_training;
	std::vector<thrust::host_vector<double>> h_val;
	std::vector<Layer> layers;
	std::vector<thrust::device_vector<double>> W;
	std::vector<thrust::device_vector<double>> B;

	void summary() {
		printf("Model Summary\n==================================================\n");
		printf("Layer (type)                    Size\n");
		printf("--------------------------------------------------\n");
		std::string layerType = "";

		for (int i = 0; i < layers.size(); i++) {
			switch (layers.at(i).type) {
			case LayerType::DENSE:
				layerType = "DENSE";
				break;
			case LayerType::INPUT:
				layerType = "INPUT";
				break;
			case LayerType::OUTPUT:
				layerType = "OUTPUT";
				break;
			}
			std::cout << "Layer_" << i << " " << layerType << "                    " << layers.at(i).size << std::endl;
		}
		printf("==================================================\n");
	}

	Model() {
		layers = std::vector<Layer>();
	}

	void add(Layer layer) {
		layers.push_back(layer);
	}

	void initWeights() {
		//Dummy 0th layer - Input layer
		W.push_back(thrust::device_vector<double>());

		for (int i = 1; i < layers.size(); i++) {

			unsigned size = layers.at(i - 1).size * layers.at(i).size;
			thrust::device_vector<double> tempWi(size);
			
			if (initializationType == InitializationType::RANDOM) {
				srand(time(0));
				thrust::counting_iterator<unsigned> iterator(0);
				thrust::transform(iterator, iterator + tempWi.size(), tempWi.begin(), prg(rand()));
			}
			else
			{
				thrust::fill(tempWi.begin(), tempWi.end(), 0.0);
			}

			W.push_back(tempWi);
		}
	}

	void initBias() {
		//Dummy 0th layer - Input layer
		B.push_back(thrust::device_vector<double>());

		for (int i = 1; i < layers.size(); i++) {
			thrust::device_vector<double> tempBi(layers.at(i).size);

			if (initializationType == InitializationType::RANDOM) {
				srand(time(0));
				thrust::counting_iterator<unsigned> iterator(0);
				thrust::transform(iterator, iterator + tempBi.size(), tempBi.begin(), prg(rand()));
			}
			else {
				thrust::fill(tempBi.begin(), tempBi.end(), 0.0);
			}

			B.push_back(tempBi);
		}
	}

	void compile(OptimizerType _optimizerType, InitializationType _initializationType, double _learningRate) {

		optimizerType = _optimizerType;
		initializationType = _initializationType;
		learningRate = _learningRate;

		initWeights();
		initBias();
	}


	void fit(std::vector<thrust::host_vector<double>> _training, std::vector<thrust::host_vector<double>> _val, unsigned _epochs, unsigned _batchSize) {
		h_training = _training;
		h_val = _val;
		epochs = _epochs;
		batchSize = _batchSize;
	}
};


int main()
{

    return 0;
}
