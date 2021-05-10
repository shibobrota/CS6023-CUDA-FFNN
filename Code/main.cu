/*
	Name : Shibobrota
	Roll : CS20M059
	Description: Cuda Accelerated Feed forward Neural Network.

	Download dataset from: https://github.com/shibobrota/CS6023-CUDA-FFNN
*/

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <cuda.h>
#include <stdio.h>
#include <vector>
#include <string>
#include <time.h>
#include <chrono>
#include <fstream>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sequence.h>
#include <thrust/transform_reduce.h>
#include <thrust/functional.h>
#include <thrust/gather.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/random.h>
#include <thrust/extrema.h>

__managed__ static bool SHOW_DEBUG_LOGS = false;

std::string TEST_DATASET_PATH = "mnist_test.csv";
std::string TRAIN_DATASET_PATH = "mnist_train.csv";

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

struct UpdateMomentumGrad
{
	double lr, gamma;

	UpdateMomentumGrad(double _gamma, double _lr) : lr(_lr), gamma(_gamma){};

	__host__ __device__ double operator()(const double &A, const double &B)
	{
		return ((gamma * A) + (lr * B));
	}
};

struct UpdateGrad
{
	double lr;

	UpdateGrad(double _lr) : lr(_lr){};

	__host__ __device__ double operator()(const double &A, const double &B)
	{
		return (A - (lr * B));
	}
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

		if (SHOW_DEBUG_LOGS)
		{
			printf("idx: %d | i: %d | j: %d\n", idx, i, j);
		}

		for (unsigned k = 0; k < n; k++)
		{
			sum += A[i * n + k] * B[k * r + j];
			if (SHOW_DEBUG_LOGS)
			{
				printf("A[i * n + k = %d]: %f | B[k * r + j = %d]: %f\n", i * n + k, A[i * n + k], k * r + j, B[k * r + j]);
			}
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
	std::vector<thrust::host_vector<double>> h_data;
	thrust::host_vector<double> h_labels;
	std::vector<Layer> layers;
	std::vector<Matrix> W;
	std::vector<Matrix> dW;
	std::vector<thrust::device_vector<double>> B;
	std::vector<thrust::device_vector<double>> dB;

	void summary()
	{
		printf("\nModel Summary\n==================================================\n");
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

		std::string optType = "";

		switch (optimizerType)
		{
		case OptimizerType::BATCH_GD:
			optType = "BATCH GRADIENT DESCENT";
			break;
		case OptimizerType::MOMENTUM_BASED_GD:
			optType = "MOMENTUM BASED GRADIENT DESCENT";
			break;
		case OptimizerType::STOCHASTIC_GD:
			optType = "STOCHASTIC GRADIENT DESCENT";
			break;
		}

		std::cout << "Optimizer: " << optType << std::endl;
		printf("Learning Rate: %f\n", learningRate);
		printf("--------------------------------------------------\n");
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

			if (SHOW_DEBUG_LOGS)
			{
				std::cout << std::endl
						  << "dW: \n";
				thrust::copy(temp_dWi.data.begin(), temp_dWi.data.end(), std::ostream_iterator<double>(std::cout, " "));
				std::cout << std::endl
						  << "W: \n";
				thrust::copy(tempWi.data.begin(), tempWi.data.end(), std::ostream_iterator<double>(std::cout, " "));
				std::cout << "\n";
			}
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

			if (SHOW_DEBUG_LOGS)
			{
				std::cout << std::endl
						  << "dB: \n";
				thrust::copy(temp_dBi.begin(), temp_dBi.end(), std::ostream_iterator<double>(std::cout, " "));
				std::cout << std::endl
						  << "W: \n";
				thrust::copy(tempBi.begin(), tempBi.end(), std::ostream_iterator<double>(std::cout, " "));
				std::cout << "\n";
			}
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

	void fit(std::pair<std::vector<thrust::host_vector<double>>, thrust::host_vector<int>> _dataset, unsigned _epochs, unsigned _batchSize)
	{
		h_data = _dataset.first;
		h_labels = _dataset.second;
		epochs = _epochs;
		batchSize = _batchSize;

		switch (optimizerType)
		{
		case OptimizerType::BATCH_GD:
			batchGD();
			break;
		case OptimizerType::MOMENTUM_BASED_GD:
			momentumGD();
			break;
		case OptimizerType::STOCHASTIC_GD:
			stochasticGD();
			break;
		default:
			printf("\nINVALID\n");
		}
	}

	void test(std::pair<std::vector<thrust::host_vector<double>>, thrust::host_vector<int>> _dataset)
	{
		h_data = _dataset.first;
		h_labels = _dataset.second;

		int accuracyCount = 0;
		double accuracy = 0.0;
		double lossSum = 0.0;
		double loss = 0.0;

		for (int i = 0; i < h_data.size(); i++)
		{

			//Copy data to Layer 0
			thrust::copy(h_data[i].begin(), h_data[i].end(), layers[0].H.begin());

			forwardProp();

			unsigned L = layers.size() - 1;
			thrust::device_vector<double>::iterator iter = thrust::max_element(layers[L].H.begin(), layers[L].H.end());
			unsigned position = iter - layers[L].H.begin();
			lossSum += *iter;
			double lossAvg = lossSum / (i + 1);
			loss = -log(lossAvg);
			if (position == h_labels[i])
			{
				accuracyCount += 1;
			}
			accuracy = (double)accuracyCount / (double)(i + 1);
			std::cout << "Accuracy: " << accuracy * 100 << " %"
					  << " Loss: " << loss << "\r";
		}
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

			if (SHOW_DEBUG_LOGS)
			{
				std::cout << std::endl
						  << "Pre Activation layer " << i << std::endl;
				thrust::copy(layers[i].A.begin(), layers[i].A.end(), std::ostream_iterator<double>(std::cout, " "));
				std::cout << std::endl
						  << "Activation layer " << i << std::endl;
				thrust::copy(layers[i].H.begin(), layers[i].H.end(), std::ostream_iterator<double>(std::cout, " "));
				std::cout << std::endl;
			}
		}
	}

	void backProp(unsigned position)
	{
		unsigned L = layers.size() - 1;
		thrust::device_vector<double> Y;

		std::vector<thrust::device_vector<double>> dA;
		std::vector<thrust::device_vector<double>> dH;

		dA.push_back(thrust::device_vector<double>());
		dH.push_back(thrust::device_vector<double>());

		for (int i = 1; i < layers.size(); i++)
		{
			dA.push_back(thrust::device_vector<double>(layers[i].size));
			if (i == L)
			{
				dH.push_back(thrust::device_vector<double>());
			}
			else
			{
				dH.push_back(thrust::device_vector<double>(layers[i].size));
			}
		}

		if (SHOW_DEBUG_LOGS)
		{
			std::cout << "\nLayer L" << std::endl;
			thrust::copy(layers[L].H.begin(), layers[L].H.end(), std::ostream_iterator<double>(std::cout, " "));
			std::cout << std::endl
					  << "Pos: " << position << std::endl;
		}

		Y = thrust::device_vector<double>(layers[L].size);

		thrust::fill(Y.begin(), Y.end(), 0.0);
		Y[position] = 1.0;

		thrust::transform(layers[L].H.begin(), layers[L].H.end(), Y.begin(), dA[L].begin(), thrust::minus<double>());

		if (SHOW_DEBUG_LOGS)
		{
			std::cout << "\n### BACK PROPAGATION ###\n"
					  << std::endl
					  << "Y" << std::endl;
			thrust::copy(Y.begin(), Y.end(), std::ostream_iterator<double>(std::cout, " "));
			std::cout << std::endl
					  << "dA[L]" << std::endl;
			thrust::copy(dA[L].begin(), dA[L].end(), std::ostream_iterator<double>(std::cout, " "));
			std::cout << std::endl;
		}

		for (int i = L; i >= 1; i--)
		{
			if (SHOW_DEBUG_LOGS)
			{
				//==========================================================================================
				std::cout << std::endl
						  << "dA[" << i << "]: " << std::endl;
				thrust::copy(dA[i].begin(), dA[i].end(), std::ostream_iterator<double>(std::cout, " "));
				std::cout << std::endl;
				std::cout << "layers[" << i - 1 << "].H: " << std::endl;
				thrust::copy(layers[i - 1].H.begin(), layers[i - 1].H.end(), std::ostream_iterator<double>(std::cout, " "));
				std::cout << std::endl;
				//==========================================================================================
			}

			matMul(dA[i], layers[i - 1].H, dW[i].data, dA[i].size(), 1, layers[i - 1].H.size());
			thrust::copy(dA[i].begin(), dA[i].end(), dB[i].begin());

			if (SHOW_DEBUG_LOGS)
			{
				//==========================================================================================
				std::cout << std::endl
						  << "dW: " << i << std::endl;
				thrust::copy(dW[i].data.begin(), dW[i].data.end(), std::ostream_iterator<double>(std::cout, " "));
				std::cout << std::endl;
				std::cout << "dB: " << i << std::endl;
				thrust::copy(dB[i].begin(), dB[i].end(), std::ostream_iterator<double>(std::cout, " "));
				std::cout << std::endl;
				//==========================================================================================
			}

			cudaDeviceSynchronize();

			if (i > 1)
			{
				thrust::device_vector<double> WiTranspose(W[i].data.size());

				matTrans(W[i].data, WiTranspose, W[i].row, W[i].col);
				matMul(WiTranspose, dA[i], dH[i - 1], W[i].col, W[i].row, 1);

				thrust::device_vector<double> gradA(layers[i - 1].size);
				layers[i - 1].getGradA(gradA);

				thrust::transform(layers[i - 1].H.begin(), layers[i - 1].H.end(), gradA.begin(), dA[i - 1].begin(), thrust::multiplies<double>());

				if (SHOW_DEBUG_LOGS)
				{
					std::cout << std::endl
							  << "Activation layer dH: " << i - 1 << std::endl;
					thrust::copy(dH[i - 1].begin(), dH[i - 1].end(), std::ostream_iterator<double>(std::cout, " "));
					std::cout << std::endl
							  << "Pre Activation layer dA: " << i - 1 << std::endl;
					thrust::copy(dA[i - 1].begin(), dA[i - 1].end(), std::ostream_iterator<double>(std::cout, " "));
					std::cout << std::endl;
				}
			}
		}
	}

	void fillZeros(std::vector<Matrix> &temp_dW, std::vector<thrust::device_vector<double>> &temp_dB)
	{
		//Initialize
		for (int j = 1; j < layers.size(); j++)
		{
			thrust::fill(temp_dB[j].begin(), temp_dB[j].end(), 0.0);
			thrust::fill(temp_dW[j].data.begin(), temp_dW[j].data.end(), 0.0);
		}
	}

	void stochasticGD()
	{

		int accuracyCount = 0;
		double accuracy = 0.0;
		double lossSum = 0.0;
		double loss = 0.0;

		std::vector<Matrix> temp_dW;
		std::vector<thrust::device_vector<double>> temp_dB;

		//Dummy 0th layer - Input layer
		temp_dB.push_back(thrust::device_vector<double>());
		temp_dW.push_back(Matrix(0, 0));

		//Initialize
		for (int j = 1; j < layers.size(); j++)
		{
			thrust::device_vector<double> temp_dBi(layers.at(j).size);
			Matrix temp_dWi = Matrix(layers.at(j).size, layers.at(j - 1).size);

			thrust::fill(temp_dBi.begin(), temp_dBi.end(), 0.0);
			thrust::fill(temp_dWi.data.begin(), temp_dWi.data.end(), 0.0);

			temp_dB.push_back(temp_dBi);
			temp_dW.push_back(temp_dWi);
		}

		for (int ep = 0; ep < epochs; ep++)
		{

			std::cout << std::endl
					  << "Epoch: " << ep + 1 << std::endl;
			accuracyCount = 0;
			lossSum = 0.0;

			for (int i = 0; i < h_data.size(); i++)
			{

				//Copy data to Layer 0
				thrust::copy(h_data[i].begin(), h_data[i].end(), layers[0].H.begin());

				forwardProp();

				unsigned L = layers.size() - 1;
				thrust::device_vector<double>::iterator iter = thrust::max_element(layers[L].H.begin(), layers[L].H.end());
				lossSum += *iter;
				unsigned position = iter - layers[L].H.begin();
				if (position == h_labels[i])
				{
					accuracyCount += 1;
				}

				backProp(h_labels[i]);

				//Accumulate
				for (int j = 1; j < layers.size(); j++)
				{
					thrust::transform(dW[j].data.begin(), dW[j].data.end(), temp_dW[j].data.begin(), temp_dW[j].data.begin(), thrust::plus<double>());
					thrust::transform(dB[j].begin(), dB[j].end(), temp_dB[j].begin(), temp_dB[j].begin(), thrust::plus<double>());
				}

				if ((i + 1) % batchSize == 0 || i == (h_data.size() - 1))
				{

					accuracy = (double)accuracyCount / (double)(i + 1);

					double lossAvg = lossSum / batchSize;

					loss = -(log(lossAvg));

					lossSum = 0.0;

					std::cout << " Accuracy: " << accuracy * 100 << " %"
							  << " Loss: " << loss << "\r";

					//Update Weights
					for (int j = 1; j < layers.size(); j++)
					{
						thrust::transform(W[j].data.begin(), W[j].data.end(), temp_dW[j].data.begin(), W[j].data.begin(), UpdateGrad(learningRate));
						thrust::transform(B[j].begin(), B[j].end(), temp_dB[j].begin(), B[j].begin(), UpdateGrad(learningRate));
					}

					fillZeros(temp_dW, temp_dB);
				}

				if (SHOW_DEBUG_LOGS)
				{
					for (int j = 1; j < layers.size(); j++)
					{
						std::cout << "W " << j << std::endl;
						thrust::copy(W[j].data.begin(), W[j].data.end(), std::ostream_iterator<double>(std::cout, " "));
						std::cout << std::endl
								  << "B " << j << std::endl;
						thrust::copy(B[j].begin(), B[j].end(), std::ostream_iterator<double>(std::cout, " "));
					}
					for (int j = 1; j < layers.size(); j++)
					{
						std::cout << "dW " << j << std::endl;
						thrust::copy(dW[j].data.begin(), dW[j].data.end(), std::ostream_iterator<double>(std::cout, " "));
						std::cout << std::endl
								  << "dB " << j << std::endl;
						thrust::copy(dB[j].begin(), dB[j].end(), std::ostream_iterator<double>(std::cout, " "));
					}
				}
			}
		}
	}

	void batchGD()
	{

		int accuracyCount = 0;
		double accuracy = 0.0;
		double lossSum = 0.0;
		double loss = 0.0;

		std::vector<Matrix> temp_dW;
		std::vector<thrust::device_vector<double>> temp_dB;

		//Dummy 0th layer - Input layer
		temp_dB.push_back(thrust::device_vector<double>());
		temp_dW.push_back(Matrix(0, 0));

		//Initialize
		for (int j = 1; j < layers.size(); j++)
		{
			thrust::device_vector<double> temp_dBi(layers.at(j).size);
			Matrix temp_dWi = Matrix(layers.at(j).size, layers.at(j - 1).size);

			thrust::fill(temp_dBi.begin(), temp_dBi.end(), 0.0);
			thrust::fill(temp_dWi.data.begin(), temp_dWi.data.end(), 0.0);

			temp_dB.push_back(temp_dBi);
			temp_dW.push_back(temp_dWi);
		}

		for (int ep = 0; ep < epochs; ep++)
		{

			std::cout << std::endl
					  << "Epoch: " << ep + 1 << std::endl;
			accuracyCount = 0;
			lossSum = 0.0;
			int i = 0;

			for (i = 0; i < h_data.size(); i++)
			{

				std::cout << "processed: " << i + 1 << "\r";

				//Copy data to Layer 0
				thrust::copy(h_data[i].begin(), h_data[i].end(), layers[0].H.begin());

				forwardProp();

				unsigned L = layers.size() - 1;
				thrust::device_vector<double>::iterator iter = thrust::max_element(layers[L].H.begin(), layers[L].H.end());
				lossSum += *iter;
				unsigned position = iter - layers[L].H.begin();
				if (position == h_labels[i])
				{
					accuracyCount += 1;
				}

				backProp(h_labels[i]);

				//Accumulate
				for (int j = 1; j < layers.size(); j++)
				{
					thrust::transform(dW[j].data.begin(), dW[j].data.end(), temp_dW[j].data.begin(), temp_dW[j].data.begin(), thrust::plus<double>());
					thrust::transform(dB[j].begin(), dB[j].end(), temp_dB[j].begin(), temp_dB[j].begin(), thrust::plus<double>());
				}
			}

			accuracy = (double)accuracyCount / (double)(i + 1);
			std::cout << std::endl
					  << " Accuracy: " << accuracy * 100 << " %";

			double lossAvg = lossSum / (double)(i + 1);

			loss = -log(lossAvg);

			std::cout << " Loss: " << loss;

			//Update Weights
			for (int j = 1; j < layers.size(); j++)
			{
				thrust::transform(W[j].data.begin(), W[j].data.end(), temp_dW[j].data.begin(), W[j].data.begin(), UpdateGrad(learningRate));
				thrust::transform(B[j].begin(), B[j].end(), temp_dB[j].begin(), B[j].begin(), UpdateGrad(learningRate));
			}

			fillZeros(temp_dW, temp_dB);
		}
	}

	void momentumGD()
	{

		int accuracyCount = 0;
		double accuracy = 0.0;
		double lossSum = 0.0;
		double loss = 0.0;
		double gamma = 0.9;

		std::vector<Matrix> temp_dW;
		std::vector<thrust::device_vector<double>> temp_dB;

		std::vector<Matrix> priv_dW;
		std::vector<thrust::device_vector<double>> priv_dB;

		//Dummy 0th layer - Input layer
		temp_dB.push_back(thrust::device_vector<double>());
		temp_dW.push_back(Matrix(0, 0));

		priv_dB.push_back(thrust::device_vector<double>());
		priv_dW.push_back(Matrix(0, 0));

		//Initialize
		for (int j = 1; j < layers.size(); j++)
		{
			thrust::device_vector<double> temp_dBi(layers.at(j).size);
			Matrix temp_dWi = Matrix(layers.at(j).size, layers.at(j - 1).size);

			thrust::device_vector<double> temp_priv_dBi(layers.at(j).size);
			Matrix temp_priv_dWi = Matrix(layers.at(j).size, layers.at(j - 1).size);

			thrust::fill(temp_dBi.begin(), temp_dBi.end(), 0.0);
			thrust::fill(temp_dWi.data.begin(), temp_dWi.data.end(), 0.0);

			thrust::fill(temp_priv_dBi.begin(), temp_priv_dBi.end(), 0.0);
			thrust::fill(temp_priv_dWi.data.begin(), temp_priv_dWi.data.end(), 0.0);

			temp_dB.push_back(temp_dBi);
			temp_dW.push_back(temp_dWi);

			priv_dB.push_back(temp_priv_dBi);
			priv_dW.push_back(temp_priv_dWi);
		}

		for (int ep = 0; ep < epochs; ep++)
		{

			std::cout << std::endl
					  << "Epoch: " << ep + 1 << std::endl;
			accuracyCount = 0;
			int i = 0;
			lossSum = 0.0;

			for (i = 0; i < h_data.size(); i++)
			{

				std::cout << "processed: " << i + 1 << "\r";

				//Copy data to Layer 0
				thrust::copy(h_data[i].begin(), h_data[i].end(), layers[0].H.begin());

				forwardProp();

				unsigned L = layers.size() - 1;
				thrust::device_vector<double>::iterator iter = thrust::max_element(layers[L].H.begin(), layers[L].H.end());
				unsigned position = iter - layers[L].H.begin();
				lossSum += *iter;
				if (position == h_labels[i])
				{
					accuracyCount += 1;
				}

				backProp(h_labels[i]);

				//Accumulate
				for (int j = 1; j < layers.size(); j++)
				{
					thrust::transform(dW[j].data.begin(), dW[j].data.end(), temp_dW[j].data.begin(), temp_dW[j].data.begin(), thrust::plus<double>());
					thrust::transform(dB[j].begin(), dB[j].end(), temp_dB[j].begin(), temp_dB[j].begin(), thrust::plus<double>());
				}
			}

			accuracy = (double)accuracyCount / (double)(i + 1);
			std::cout << std::endl
					  << " Accuracy: " << accuracy * 100 << " %";

			double lossAvg = lossSum / (double)(i + 1);

			loss = -(log(lossAvg));

			std::cout << " Loss: " << loss;

			//Update Weights
			for (int j = 1; j < layers.size(); j++)
			{
				thrust::device_vector<double> v_dBi(layers.at(j).size);
				Matrix v_dWi = Matrix(layers.at(j).size, layers.at(j - 1).size);

				thrust::transform(priv_dB[j].begin(), priv_dB[j].end(), temp_dB[j].begin(), v_dBi.begin(), UpdateMomentumGrad(gamma, learningRate));
				thrust::transform(priv_dW[j].data.begin(), priv_dW[j].data.end(), temp_dW[j].data.begin(), v_dWi.data.begin(), UpdateMomentumGrad(gamma, learningRate));

				thrust::transform(W[j].data.begin(), W[j].data.end(), v_dWi.data.begin(), W[j].data.begin(), thrust::minus<double>());
				thrust::transform(B[j].begin(), B[j].end(), v_dBi.begin(), B[j].begin(), thrust::minus<double>());

				thrust::copy(v_dBi.begin(), v_dBi.end(), priv_dB[j].begin());
				thrust::copy(v_dWi.data.begin(), v_dWi.data.end(), priv_dW[j].data.begin());
			}

			fillZeros(temp_dW, temp_dB);
		}
	}
};

std::pair<std::vector<thrust::host_vector<double>>, thrust::host_vector<int>> readDataSet(std::string path)
{

	std::cout << "\nReading Dataset: " << path << std::endl;

	std::vector<thrust::host_vector<double>> dataset;
	std::ifstream fin;
	fin.open(path);

	std::string num, temp;
	thrust::host_vector<int> host_labels;

	int countLines = 0;
	std::string str = "|||///---\\\\\\|||///---\\\\\\";
	char loader = str[0];

	while (fin >> temp)
	{

		countLines += 1;
		if (countLines % 20 == 0)
			loader = str[countLines % str.length()];
		std::cout << "Number of Lines Read: " << countLines << " " << loader << "\r";

		//To break
		std::stringstream s(temp);

		thrust::host_vector<double> row;
		unsigned i = 0;
		//Read col
		while (getline(s, num, ','))
		{
			if (i == 0)
			{
				host_labels.push_back(std::stoi(num));
			}
			else
			{
				row.push_back(stod(num) / (double)255);
			}
			i++;
		}

		if (SHOW_DEBUG_LOGS)
		{
			std::cout << std::endl;
			thrust::copy(row.begin(), row.end(), std::ostream_iterator<double>(std::cout, " "));
			std::cout << std::endl;
		}

		thrust::device_vector<double> dvec = row;
		dataset.push_back(dvec);
	}

	std::pair<std::vector<thrust::host_vector<double>>, thrust::host_vector<int>> ret = std::make_pair(dataset, host_labels);

	std::cout << std::endl
			  << "Dataset read!" << std::endl;

	return ret;
}

int main()
{

	thrust::host_vector<int> labels;

	std::pair<std::vector<thrust::host_vector<double>>, thrust::host_vector<int>> trainDataset = readDataSet(TRAIN_DATASET_PATH);

	unsigned inputSize = trainDataset.first[0].size(), hiddenSize = 100, outputSize = 10;

	Model model;

	model.add(Layer(LayerType::INPUT, inputSize, ActivationType::NONE));
	model.add(Layer(LayerType::DENSE, hiddenSize, ActivationType::TANH));
	model.add(Layer(LayerType::DENSE, hiddenSize, ActivationType::TANH));
	model.add(Layer(LayerType::DENSE, hiddenSize, ActivationType::TANH));
	model.add(Layer(LayerType::DENSE, hiddenSize, ActivationType::TANH));
	model.add(Layer(LayerType::OUTPUT, outputSize, ActivationType::SOFTMAX));

	model.compile(OptimizerType::STOCHASTIC_GD, InitializationType::RANDOM, 0.001);

	model.summary();

	std::cout << std::endl
			  << "\nTraining Model\n"
			  << std::endl;

	model.fit(trainDataset, 3, 16);

	std::pair<std::vector<thrust::host_vector<double>>, thrust::host_vector<int>> testDataset = readDataSet(TEST_DATASET_PATH);

	std::cout << std::endl
			  << "\nTesting Model\n"
			  << std::endl;

	model.test(testDataset);

	return 0;
}
