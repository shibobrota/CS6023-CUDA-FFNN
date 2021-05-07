
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <vector>
#include <string>
#include <time.h>

#include <cublas_v2.h>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sequence.h>
#include <thrust/transform_reduce.h>
#include <thrust/functional.h>
#include <thrust/gather.h>
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

struct matrixMult
{
	double* A, * B;
	int m, n, r;
	
	matrixMult(double* _A, double* _B, unsigned _m, unsigned _n, unsigned _r) : A(_A), B(_B), m(_m), n(_n), r(_r) {};

	__host__ __device__
	double operator()(unsigned idx) {
		unsigned i = idx / r;
		unsigned j = idx % r;
		double sum = 0.0;

		for (unsigned k = 0; k < n; k++) {
			sum += A[i * r + k] * B[k * r + j];
		}
		return sum;
	}
};

struct transposeIndex
{
	unsigned m, n;

	__host__ __device__
		transposeIndex(unsigned _m, unsigned _n) : m(_m), n(_n) {}

	__host__ __device__
	unsigned operator()(unsigned linear_index)
	{
		unsigned i = linear_index / n;
		unsigned j = linear_index % n;

		return (m * j + i);
	}
};

struct prg
{
	double a, b;
	thrust::default_random_engine rng;
	thrust::uniform_real_distribution<double> dist;

	__host__ __device__
	prg(unsigned seed, double _a = 0.0, double _b = 1.0) : a(_a), b(_b)
	{
		this->rng = thrust::default_random_engine(seed);
		this->dist = thrust::uniform_real_distribution<double>(a, b);
	};

	__host__ __device__ double operator()(unsigned d)
	{
		this->rng.discard(d);
		return this->dist(rng);
	}
};

struct Exp
{
	__host__ __device__ double operator()(const double &x) const
	{
		return exp(x);
	}
};

struct ReLU
{

	/* Unary Operator */
	__host__ __device__ double operator()(double data)
	{
		return max(data, 0.0);
	}
};

struct TanH
{
	__host__ __device__ double operator()(double data)
	{
		return tanh(data);
	}
};

struct Sigmoid
{
	__host__ __device__ double operator()(double data)
	{
		return (1.0 / (1 + exp(-data)));
	}
};

struct Softmax
{
	double sum;

	Softmax(double _sum) : sum(_sum) {}

	__host__ __device__ double operator()(double data)
	{
		return (exp(data) / sum);
	}
};

class Matrix {
public:
	unsigned row, col;
	thrust::device_vector<double> data;

	Matrix(unsigned _row, unsigned _col) {
		row = _row;
		col = _col;

		data = thrust::device_vector<double>(row * col);
	}
};

class Layer
{

public:
	LayerType type;
	unsigned size;
	ActivationType activation;

	/* Activation Layer Nodes */
	thrust::device_vector<double> d_H;
	thrust::host_vector<double> h_H;

	/* Pre-Activation Layer Nodes */
	thrust::device_vector<double> d_A;

	void applyActivation()
	{

		switch (activation)
		{
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

	Layer(LayerType _type, unsigned _size, ActivationType _activation)
	{
		size = _size;
		activation = _activation;
		type = _type;

		if (type == LayerType::INPUT)
		{
			h_H = thrust::host_vector<double>(size);
		}
		else
		{
			d_A = thrust::device_vector<double>(size);
		}
		d_H = thrust::device_vector<double>(size);
	}
};

class Model
{

	bool isValid()
	{
		if (layers.at(0).type != LayerType::INPUT || layers.at(layers.size() - 1).type != LayerType::OUTPUT)
		{
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
	std::vector<Matrix> W;
	std::vector<thrust::device_vector<double>> B;

	void summary()
	{
		printf("Model Summary\n==================================================\n");
		printf("Layer (type)                    Size\n");
		printf("--------------------------------------------------\n");
		std::string layerType = "";

		for (int i = 0; i < layers.size(); i++)
		{
			switch (layers.at(i).type)
			{
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

	Model()
	{
		layers = std::vector<Layer>();
	}

	void add(Layer layer)
	{
		layers.push_back(layer);
	}

	void initWeights()
	{
		//Dummy 0th layer - Input layer
		Matrix temp = Matrix(0, 0);
		W.push_back(temp);

		for (int i = 1; i < layers.size(); i++)
		{

			Matrix tempWi = Matrix(layers.at(i).size, layers.at(i - 1).size);

			if (initializationType == InitializationType::RANDOM)
			{
				srand(time(0));
				thrust::counting_iterator<unsigned> iterator(0);
				thrust::transform(iterator, iterator + tempWi.data.size(), tempWi.data.begin(), prg(rand()));
			}
			else
			{
				thrust::fill(tempWi.data.begin(), tempWi.data.end(), 0.0);
			}

			W.push_back(tempWi);
		}
	}

	void initBias()
	{
		//Dummy 0th layer - Input layer
		B.push_back(thrust::device_vector<double>());

		for (int i = 1; i < layers.size(); i++)
		{
			thrust::device_vector<double> tempBi(layers.at(i).size);

			if (initializationType == InitializationType::RANDOM)
			{
				srand(time(0));
				thrust::counting_iterator<unsigned> iterator(0);
				thrust::transform(iterator, iterator + tempBi.size(), tempBi.begin(), prg(rand()));
			}
			else
			{
				thrust::fill(tempBi.begin(), tempBi.end(), 0.0);
			}

			B.push_back(tempBi);
		}
	}

	void compile(OptimizerType _optimizerType, InitializationType _initializationType, double _learningRate)
	{

		optimizerType = _optimizerType;
		initializationType = _initializationType;
		learningRate = _learningRate;

		if (!isValid())
		{
			std::cout << "Unable to Compile Model ---- Please add Input/Output Layer(s)" << std::endl;
			return;
		}

		initWeights();
		initBias();
	}

	void fit(std::vector<thrust::host_vector<double>> _training, std::vector<thrust::host_vector<double>> _val, unsigned _epochs, unsigned _batchSize)
	{
		h_training = _training;
		h_val = _val;
		epochs = _epochs;
		batchSize = _batchSize;
	}

	void matMul(thrust::device_vector<double> A, thrust::device_vector<double> B, thrust::device_vector<double> &C, 
			unsigned m, unsigned n, unsigned r) {

		thrust::counting_iterator<unsigned> iter(0);
		thrust::transform(iter, iter+(m*r), C.begin(), matrixMult(thrust::raw_pointer_cast(A.data()), thrust::raw_pointer_cast(B.data()), m, n, r));
	}

	void matTrans(thrust::device_vector<double> A, thrust::device_vector<double>& B, unsigned m, unsigned n) {
		thrust::counting_iterator<size_t> iter(0);
		thrust::gather(thrust::make_transform_iterator(iter, transposeIndex(n, m)), thrust::make_transform_iterator(iter, transposeIndex(n, m)) + B.size(), A.begin(), B.begin());
	}

	void matAdd(thrust::device_vector<double> A, thrust::device_vector<double> B, thrust::device_vector<double>& C, unsigned m, unsigned n) {
		thrust::transform(A.begin(), A.end(), B.begin(), C.begin(), thrust::plus<double>());
	}

	void forwardProp() {
		//
	}
};

int main()
{
	unsigned m = 10, n = 20, r = 5;
	thrust::device_vector<double > A(m * n);
	thrust::device_vector<double > B(m * n);
	thrust::device_vector<double > C(m * n);

	thrust::fill(A.begin(), A.end(), 1.8);
	thrust::fill(B.begin(), B.end(), 5.2);

	thrust::copy(A.begin(), A.end(), std::ostream_iterator<double>(std::cout, " "));
	std::cout << std::endl;
	thrust::copy(B.begin(), B.end(), std::ostream_iterator<double>(std::cout, " "));
	std::cout << std::endl;

	Model model;

	//model.matMul(A, B, C, m, n, r);
	model.matAdd(A, B, C, m, n);
	cudaDeviceSynchronize();

	thrust::copy(C.begin(), C.end(), std::ostream_iterator<double>(std::cout, " "));

	std::cout << "No Error!!";
	return 0;
}
