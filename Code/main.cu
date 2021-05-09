﻿
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <vector>
#include <string>
#include <time.h>
#include <chrono>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sequence.h>
#include <thrust/transform_reduce.h>
#include <thrust/functional.h>
#include <thrust/gather.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/random.h>
#include <thrust/extrema.h>

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
	double *A, *B;
	int m, n, r;

	matrixMult(double *_A, double *_B, unsigned _m, unsigned _n, unsigned _r) : A(_A), B(_B), m(_m), n(_n), r(_r){};

	__host__ __device__ double operator()(unsigned idx)
	{
		unsigned i = idx / r;
		unsigned j = idx % r;
		double sum = 0.0;

		printf("idx: %d | i: %d | j: %d\n", idx, i, j);

		for (unsigned k = 0; k < n; k++)
		{
			sum += A[i * n + k] * B[k * r + j];
			printf("A[i * n + k = %d]: %f | B[k * r + j = %d]: %f\n", i * n + k, A[i * n + k], k * r + j, B[k * r + j]);
		}
		return sum;
	}
};

struct transposeIndex
{
	unsigned m, n;

	__host__ __device__
	transposeIndex(unsigned _m, unsigned _n) : m(_m), n(_n) {}

	__host__ __device__ unsigned operator()(unsigned linear_index)
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
	prg(unsigned seed, double _a = -1.0, double _b = 1.0) : a(_a), b(_b)
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

struct GradReLU
{

	/* Unary Operator */
	__host__ __device__ double operator()(double data)
	{
		return (data > 0.0) ? 1.0 : 0.0;
	}
};

struct TanH
{
	__host__ __device__ double operator()(double data)
	{
		return tanh(data);
	}
};

struct GradTanH
{
	__host__ __device__ double operator()(double data)
	{
		return (1 - pow(tanh(data), 2));
	}
};

struct Sigmoid
{
	__host__ __device__ double operator()(double data)
	{
		return (1.0 / (1 + exp(-data)));
	}
};

struct GradSigmoid
{
	__host__ __device__ double operator()(double data)
	{
		double sigX = (1.0 / (1 + exp(-data)));
		return sigX * (1 - sigX);
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

class Matrix
{
public:
	unsigned row, col;
	thrust::device_vector<double> data;

	Matrix(unsigned _row, unsigned _col)
	{
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
	thrust::device_vector<double> H;

	/* Pre-Activation Layer Nodes */
	thrust::device_vector<double> A;

	Layer(LayerType _type, unsigned _size, ActivationType _activation)
	{
		size = _size;
		activation = _activation;
		std::cout << activation;
		type = _type;

		if (type != LayerType::INPUT)
		{
			A = thrust::device_vector<double>(size);
		}
		H = thrust::device_vector<double>(size);
	}

	void applyActivation()
	{
		switch (activation)
		{
		case ActivationType::SOFTMAX:
		{
			double sum = thrust::transform_reduce(A.begin(), A.end(), Exp(), 0.0, thrust::plus<float>());
			thrust::transform(A.begin(), A.end(), H.begin(), Softmax(sum));
			break;
		}
		case ActivationType::RELU:
		{
			thrust::transform(A.begin(), A.end(), H.begin(), ReLU());
			break;
		}
		case ActivationType::TANH:
		{
			thrust::transform(A.begin(), A.end(), H.begin(), TanH());
			break;
		}
		case ActivationType::SIGMOID:
		{
			thrust::transform(A.begin(), A.end(), H.begin(), Sigmoid());
			break;
		}
		default:
		{
			printf("\nUndefined Activation!\n");
		}
		}
	}

	void getGradA(thrust::device_vector<double> &t)
	{
		switch (activation)
		{
		case ActivationType::RELU:
		{
			thrust::transform(A.begin(), A.end(), t.begin(), GradReLU());
			break;
		}
		case ActivationType::SIGMOID:
		{
			thrust::transform(A.begin(), A.end(), t.begin(), GradSigmoid());
			break;
		}
		case ActivationType::TANH:
		{
			thrust::transform(A.begin(), A.end(), t.begin(), GradTanH());
			break;
		}
		}

		cudaDeviceSynchronize();
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
	std::vector<Matrix> dW;
	std::vector<thrust::device_vector<double>> B;
	std::vector<thrust::device_vector<double>> dB;

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
		Matrix temp = Matrix(0, 0), temp_dW = Matrix(0, 0);
		W.push_back(temp);
		dW.push_back(temp_dW);

		for (int i = 1; i < layers.size(); i++)
		{

			Matrix tempWi = Matrix(layers.at(i).size, layers.at(i - 1).size);
			Matrix temp_dWi = Matrix(layers.at(i).size, layers.at(i - 1).size);

			if (initializationType == InitializationType::RANDOM)
			{
				srand(std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count());
				thrust::counting_iterator<unsigned> iterator(0);
				thrust::transform(iterator, iterator + tempWi.data.size(), tempWi.data.begin(), prg(rand()));
			}
			else
			{
				thrust::fill(tempWi.data.begin(), tempWi.data.end(), 0.0);
			}

			thrust::fill(temp_dWi.data.begin(), temp_dWi.data.end(), 0.0);

			dW.push_back(temp_dWi);
			W.push_back(tempWi);

			std::cout << std::endl
					  << "dW: \n";
			thrust::copy(temp_dWi.data.begin(), temp_dWi.data.end(), std::ostream_iterator<double>(std::cout, " "));
			std::cout << std::endl
					  << "W: \n";
			thrust::copy(tempWi.data.begin(), tempWi.data.end(), std::ostream_iterator<double>(std::cout, " "));
			std::cout << "\n";
		}
	}

	void initBias()
	{
		//Dummy 0th layer - Input layer
		B.push_back(thrust::device_vector<double>());
		dB.push_back(thrust::device_vector<double>());

		for (unsigned i = 1; i < layers.size(); i++)
		{
			thrust::device_vector<double> tempBi(layers.at(i).size);
			thrust::device_vector<double> temp_dBi(layers.at(i).size);

			if (initializationType == InitializationType::RANDOM)
			{
				srand(std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count());
				thrust::counting_iterator<unsigned> iterator(0);
				thrust::transform(iterator, iterator + tempBi.size(), tempBi.begin(), prg(rand()));
			}
			else
			{
				thrust::fill(tempBi.begin(), tempBi.end(), 0.0);
			}

			thrust::fill(temp_dBi.begin(), temp_dBi.end(), 0.0);

			B.push_back(tempBi);
			dB.push_back(temp_dBi);

			std::cout << std::endl
					  << "dB: \n";
			thrust::copy(temp_dBi.begin(), temp_dBi.end(), std::ostream_iterator<double>(std::cout, " "));
			std::cout << std::endl
					  << "W: \n";
			thrust::copy(tempBi.begin(), tempBi.end(), std::ostream_iterator<double>(std::cout, " "));
			std::cout << "\n";
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

		layers[0].H = h_training[0];

		forwardProp();
		cudaDeviceSynchronize();
		backProp();
	}

	void matMul(thrust::device_vector<double> A, thrust::device_vector<double> B, thrust::device_vector<double> &C,
				unsigned m, unsigned n, unsigned r)
	{

		thrust::counting_iterator<unsigned> iter(0);
		thrust::transform(iter, iter + (m * r), C.begin(), matrixMult(thrust::raw_pointer_cast(A.data()), thrust::raw_pointer_cast(B.data()), m, n, r));
	}

	void matTrans(thrust::device_vector<double> A, thrust::device_vector<double> &B, unsigned m, unsigned n)
	{
		thrust::counting_iterator<size_t> iter(0);
		thrust::gather(thrust::make_transform_iterator(iter, transposeIndex(n, m)), thrust::make_transform_iterator(iter, transposeIndex(n, m)) + B.size(), A.begin(), B.begin());
	}

	void forwardProp()
	{
		for (unsigned i = 1; i < layers.size(); i++)
		{
			thrust::device_vector<double> res(layers[i].size);

			matMul(W[i].data, layers[i - 1].H, res, W[i].row, W[i].col, 1);
			thrust::transform(res.begin(), res.end(), B[i].begin(), layers[i].A.begin(), thrust::plus<double>());
			layers[i].applyActivation();

			std::cout << std::endl
					  << "Pre Activation layer " << i << std::endl;
			thrust::copy(layers[i].A.begin(), layers[i].A.end(), std::ostream_iterator<double>(std::cout, " "));
			std::cout << std::endl
					  << "Activation layer " << i << std::endl;
			thrust::copy(layers[i].H.begin(), layers[i].H.end(), std::ostream_iterator<double>(std::cout, " "));
			std::cout << std::endl;
		}
	}

	void backProp()
	{
		unsigned L = layers.size() - 1;
		thrust::device_vector<double> Y;

		std::vector<thrust::device_vector<double>> dA;
		std::vector<thrust::device_vector<double>> dH;

		dA.push_back(thrust::device_vector<double>());
		dH.push_back(thrust::device_vector<double>());

		for (int i = 1; i < layers.size(); i++) {
			dA.push_back(thrust::device_vector<double>(layers[i].size));
			if (i == L) {
				dH.push_back(thrust::device_vector<double>());
			}
			else {
				dH.push_back(thrust::device_vector<double>(layers[i].size));
			}
		}

		thrust::device_vector<double>::iterator iter = thrust::max_element(layers[L].H.begin(), layers[L].H.end());
		unsigned position = iter - layers[L].H.begin();

		std::cout << "\nLayer L" << std::endl;
		thrust::copy(layers[L].H.begin(), layers[L].H.end(), std::ostream_iterator<double>(std::cout, " "));
		std::cout << std::endl
				  << "Pos: " << position << std::endl;

		Y = thrust::device_vector<double>(layers[L].size);

		thrust::fill(Y.begin(), Y.end(), 0.0);
		Y[position] = 1.0;


		thrust::transform(layers[L].H.begin(), layers[L].H.end(), Y.begin(), dA[L].begin(), thrust::minus<double>());

		std::cout << "\n### BACK PROPAGATION ###\n"
				  << std::endl
				  << "Y" << std::endl;
		thrust::copy(Y.begin(), Y.end(), std::ostream_iterator<double>(std::cout, " "));
		std::cout << std::endl
				  << "dA[L]" << std::endl;
		thrust::copy(dA[L].begin(), dA[L].end(), std::ostream_iterator<double>(std::cout, " "));
		std::cout << std::endl;

		for (int i = L; i >= 1; i--)
		{
			//==========================================================================================
			std::cout << std::endl
				<< "dA[" << i <<"]: " << std::endl;
			thrust::copy(dA[i].begin(), dA[i].end(), std::ostream_iterator<double>(std::cout, " "));
			std::cout << std::endl;
			std::cout << "layers[" << i-1 <<"].H: "<< std::endl;
			thrust::copy(layers[i - 1].H.begin(), layers[i - 1].H.end(), std::ostream_iterator<double>(std::cout, " "));
			std::cout << std::endl;
			//==========================================================================================


			matMul(dA[i], layers[i - 1].H, dW[i].data, dA[i].size(), 1, layers[i - 1].H.size());
			thrust::copy(dA[i].begin(), dA[i].end(), dB[i].begin());


			//==========================================================================================
			std::cout << std::endl
					  << "dW: " << i << std::endl;
			thrust::copy(dW[i].data.begin(), dW[i].data.end(), std::ostream_iterator<double>(std::cout, " "));
			std::cout << std::endl;
			std::cout << "dB: " << i << std::endl;
			thrust::copy(dB[i].begin(), dB[i].end(), std::ostream_iterator<double>(std::cout, " "));
			std::cout << std::endl;
			//==========================================================================================

			cudaDeviceSynchronize();

			if (i > 1)
			{
				thrust::device_vector<double> WiTranspose(W[i].data.size());

				matTrans(W[i].data, WiTranspose, W[i].row, W[i].col);
				matMul(WiTranspose, dA[i], dH[i-1], W[i].col, W[i].row, 1);

				thrust::device_vector<double> gradA(layers[i-1].size);
				layers[i - 1].getGradA(gradA);

				thrust::transform(layers[i - 1].H.begin(), layers[i - 1].H.end(), gradA.begin(), dA[i-1].begin(), thrust::multiplies<double>());

				std::cout << std::endl
						  << "Activation layer dH: " << i-1 << std::endl;
				thrust::copy(dH[i-1].begin(), dH[i - 1].end(), std::ostream_iterator<double>(std::cout, " "));
				std::cout << std::endl
						  << "Pre Activation layer dA: " << i-1 << std::endl;
				thrust::copy(dA[i - 1].begin(), dA[i - 1].end(), std::ostream_iterator<double>(std::cout, " "));
				std::cout << std::endl;
			}
		}
	}
};

int main()
{
	unsigned inputSize = 10, hiddenSize = 5, outputSize = 2;
	thrust::host_vector<double> A(inputSize);

	thrust::fill(A.begin(), A.end(), 0.021);

	Model model;

	model.add(Layer(LayerType::INPUT, inputSize, ActivationType::NONE));
	model.add(Layer(LayerType::DENSE, hiddenSize, ActivationType::SIGMOID));
	model.add(Layer(LayerType::DENSE, hiddenSize, ActivationType::SIGMOID));
	model.add(Layer(LayerType::OUTPUT, outputSize, ActivationType::SOFTMAX));

	model.compile(OptimizerType::BATCH_GD, InitializationType::RANDOM, 0.001);

	std::vector<thrust::host_vector<double>> dataset;
	dataset.push_back(A);

	model.fit(dataset, dataset, 1, 1);

	return 0;
}
