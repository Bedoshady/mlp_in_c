#include<iostream>
#include<math.h>
#include"h5_loader.h"
//#include"mnist_loader.h"

enum activation_function {
	sigmoid,
	relu,
	softmax
};

struct node {
	float Value;
	float Bias;
	float IntermediateValue;
};

struct layer {
	node* Nodes;
	int NumberOfNodes;

	activation_function ActivationFunction;
};

struct mlp {
	layer* Layers;
	int NumberOfLayers;
	weight_matrix* Matrices;
	int NumberOfMatrices;
};

void InitializeMLP(mlp* Mlp, int NodesPerInput, int NodesPerHiddenLayer,
	int NumberOfHiddenLayers, int NodesPerOutput) {
	Mlp->NumberOfLayers = 2 + NumberOfHiddenLayers;
	Mlp->Layers = new layer[Mlp->NumberOfLayers];

	Mlp->Layers[0].NumberOfNodes = NodesPerInput;
	Mlp->Layers[0].Nodes = new node[Mlp->Layers[0].NumberOfNodes];
	Mlp->Layers[0].ActivationFunction = sigmoid;

	for (int LayerIndex = 1; LayerIndex < NumberOfHiddenLayers + 1; LayerIndex++) {
		Mlp->Layers[LayerIndex].NumberOfNodes = NodesPerHiddenLayer;
		Mlp->Layers[LayerIndex].Nodes = new node[Mlp->Layers[LayerIndex].NumberOfNodes];

		Mlp->Layers[LayerIndex].ActivationFunction = sigmoid;
	}
	Mlp->Layers[Mlp->NumberOfLayers - 1].NumberOfNodes = NodesPerOutput;
	Mlp->Layers[Mlp->NumberOfLayers - 1].Nodes = new node[Mlp->Layers[Mlp->NumberOfLayers - 1].NumberOfNodes];
	Mlp->Layers[Mlp->NumberOfLayers - 1].ActivationFunction = softmax;

	Mlp->NumberOfMatrices = Mlp->NumberOfLayers - 1;
	Mlp->Matrices = new weight_matrix[Mlp->NumberOfMatrices];

	
	for (int MatrixIndex = 0; MatrixIndex < Mlp->NumberOfMatrices; MatrixIndex++) {
		Mlp->Matrices[MatrixIndex].NumberOfColumns = Mlp->Layers[MatrixIndex].NumberOfNodes;
		Mlp->Matrices[MatrixIndex].NumberOfRows = Mlp->Layers[MatrixIndex + 1].NumberOfNodes;
		Mlp->Matrices[MatrixIndex].Weights = new float[Mlp->Matrices[MatrixIndex].NumberOfColumns * Mlp->Matrices[MatrixIndex].NumberOfRows];
	}
}

void RandomizeMlp(mlp* Mlp) {
	for (int LayerIndex = 0; LayerIndex < Mlp->NumberOfLayers - 1; LayerIndex++) {
		for (int NodeIndex = 0; NodeIndex < Mlp->Layers[LayerIndex + 1].NumberOfNodes; NodeIndex++) {
			Mlp->Layers[LayerIndex + 1].Nodes[NodeIndex].Bias = 2.0 * rand() / RAND_MAX - 1;
		}
		for (int WeightIndex = 0; WeightIndex < Mlp->Matrices[LayerIndex].NumberOfRows *
			Mlp->Matrices[LayerIndex].NumberOfColumns; WeightIndex++) {
			Mlp->Matrices[LayerIndex].Weights[WeightIndex] = 2.0 * rand() / RAND_MAX - 1;
		}
	
	}

}



float ApplyActivationFunction(float Input, activation_function Function) {
	switch (Function) {
	case sigmoid: {
		return 1.0f / (1.0f + exp(-Input));
	}break;
	case relu: {
		if (Input < 0)
			return 0;
		else
			return Input;
	}break;
	}

	std::cout << "Unknown activation function";
	return 1000;

}

void MatrixMultiply(weight_matrix* Matrix, layer* Input, layer* Output, bool IsOutputLayer) {
	float Total = 0;
	for (int RowIndex = 0; RowIndex < Matrix->NumberOfRows; RowIndex++) {
		Output->Nodes[RowIndex].Value = 0;
		for (int ColIndex = 0; ColIndex < Matrix->NumberOfColumns; ColIndex++) {
			Output->Nodes[RowIndex].Value += Input->Nodes[ColIndex].Value * 
				Matrix->Weights[RowIndex * Matrix->NumberOfColumns + ColIndex];
		}
		Output->Nodes[RowIndex].Value += Output->Nodes[RowIndex].Bias;
		Output->Nodes[RowIndex].IntermediateValue = Output->Nodes[RowIndex].Value;

		if(!IsOutputLayer || Output->ActivationFunction != softmax)
			Output->Nodes[RowIndex].Value = ApplyActivationFunction(Output->Nodes[RowIndex].Value, Output->ActivationFunction);
		else
			Total += exp(Output->Nodes[RowIndex].Value);
	}
	if (IsOutputLayer && Output->ActivationFunction == softmax) {
		for (int NodeIndex = 0; NodeIndex < Output->NumberOfNodes; NodeIndex++) {
			Output->Nodes[NodeIndex].Value = exp(Output->Nodes[NodeIndex].Value) / Total;
		}
	}
}


void FarwardPass(mlp* Mlp) {
	for (int LayerIndex = 0; LayerIndex < Mlp->NumberOfLayers - 1; LayerIndex++) {
		MatrixMultiply(&Mlp->Matrices[LayerIndex], &Mlp->Layers[LayerIndex], &Mlp->Layers[LayerIndex + 1],
			LayerIndex + 1 == Mlp->NumberOfLayers - 1);
	}
}

/*
void tom() {
	mlp Mlp;
	InitializeMLP(&Mlp, 2, 8, 2, 2);

	float Weights[] = { -0.289758f, -0.341226f , 1.443016f, -1.294722f ,
	-1.199916f, -1.235250f,
	0.946683f, 0.788452f,
	0.302699f, 0.209988f,
	1.030959f, 0.869180f,
	0.149584f, -0.129121f,
	0.679692f, 0.972907f };
	for (int WeightIndex = 0; WeightIndex < 16; WeightIndex++) {
		Mlp.Matrices[0].Weights[WeightIndex] = Weights[WeightIndex];
	}

	float InputBiases[] = { -0.352048f, -0.450067f, 1.223930f, -0.783500f, -0.671987f, -0.898360f, -0.676135f, 0.832687f };
	for (int BiasIndex = 0; BiasIndex < 8; BiasIndex++) {
		Mlp.Layers[1].Nodes[BiasIndex].Bias = InputBiases[BiasIndex];
	}

	float HiddenLayer[] = {
	0.164999f, -0.216283f, 0.267871f, 0.083952f, 0.086033f, -0.185758f, 0.265139f, -0.106266f,
	0.058458f, 0.458599f, -1.138265f, -1.249847f, -0.322268f, -1.103818f, 0.141496f, 0.967665f,
	0.052247f, 1.435651f, -1.633166f, -0.553701f, -0.093960f, -1.124455f, -0.266096f, 1.097304f,
	0.229468f, -0.459296f, 1.602662f, 1.239617f, 0.297255f, 1.716253f, -0.134797f, 0.333224f,
	-0.178098f, -0.111790f, -0.046482f, -0.156621f, -0.124190f, -0.314205f, -0.298897f, -0.015613f,
	0.235458f, -0.223764f, 1.773395f, 1.503393f, 0.350309f, 1.497746f, 0.001594f, 0.298486f,
	0.194882f, -0.371485f, -0.132792f, -0.105630f, -0.315042f, 0.302747f, -0.256041f, -0.233207f,
	-0.104823f, -0.291684f, -0.253147f, 0.213346f, -0.091313f, -0.032163f, -0.326580f, -0.092474f
	};
	for (int WeightIndex = 0; WeightIndex < 64; WeightIndex++) {
		Mlp.Matrices[1].Weights[WeightIndex] = HiddenLayer[WeightIndex];
	}

	float HiddenBiases[] = { -0.274183f, 0.425024f, 0.974474f, -0.147347f, -0.076183f, -0.191622f, -0.298609f, -0.270596f };
	for (int BiasIndex = 0; BiasIndex < 8; BiasIndex++) {
		Mlp.Layers[2].Nodes[BiasIndex].Bias = HiddenBiases[BiasIndex];
	}

	float OutputLayer[] = {
	-0.107642f, -1.142470f, -1.374395f, 1.812516f, 0.210806f, 1.780124f, 0.018185f, -0.247536f,
	0.275798f, 0.934246f, 1.696588f, -1.516321f, -0.027512f, -1.548799f, -0.190009f, -0.289626f
	};
	for (int WeightIndex = 0; WeightIndex < 16; WeightIndex++) {
		Mlp.Matrices[2].Weights[WeightIndex] = OutputLayer[WeightIndex];
	}

	float OutputBias[] = { -0.871736f, 0.837122f };
	for (int BiasIndex = 0; BiasIndex < 2; BiasIndex++) {
		Mlp.Layers[3].Nodes[BiasIndex].Bias = OutputBias[BiasIndex];
	}
	

	Mlp.Layers[0].Nodes[0].Value = 0;
	Mlp.Layers[0].Nodes[1].Value = 1;

	FarwardPass(&Mlp);

	std::cout << Mlp.Layers[3].Nodes[0].Value << std::endl;
	std::cout << Mlp.Layers[3].Nodes[1].Value;
}
*/


void LoadH5ToMlp(mlp* Mlp, const char* FileName) {
	mlp_information Result = LoadH5File(FileName);
	for (int LayerIndex = 0; LayerIndex < Result.NumberOfLayers; LayerIndex++) {
		for (int NodeIndex = 0; NodeIndex < Mlp->Layers[LayerIndex + 1].NumberOfNodes; NodeIndex++) {

			Mlp->Layers[LayerIndex + 1].Nodes[NodeIndex].Bias = Result.LayerBiases[LayerIndex].Biases[NodeIndex];
		}
		free(Result.LayerBiases[LayerIndex].Biases);
		free(Mlp->Matrices[LayerIndex].Weights);
		Mlp->Matrices[LayerIndex].Weights = Result.MlpMatrices[LayerIndex].Weights;
	}

}


struct layer_deratives {
	float* Deratives;
	int NumberOfNodes;
};

float ComputeDerivativeOfActivationFunction(float InputValue, activation_function Function) {
	switch (Function)
	{
	case sigmoid: {
		return InputValue * (1 - InputValue);
	}break;
	case relu: {
		if (InputValue < 0.0001)
			return 0;
		else
			return 1;
	}break;
	case softmax: {
		return 1;
	}break;
	}
	std::cout << "Unknown activation function";
	return -1000;
}

void TrainMlp(mlp* Mlp, layer* InputLayers, layer* ExpectedOutputs, int NumberOfExamples) {
	layer_deratives* LayerDeratives = new layer_deratives[Mlp->NumberOfLayers];
	weight_matrix* WeightDeratives = new weight_matrix[Mlp->NumberOfLayers - 1];
	
	layer_deratives* BiasDeratives = new layer_deratives[Mlp->NumberOfLayers];

	for (int LayerIndex = 0; LayerIndex < Mlp->NumberOfLayers; LayerIndex++) {
		LayerDeratives[LayerIndex].NumberOfNodes = Mlp->Layers[LayerIndex].NumberOfNodes;
		LayerDeratives[LayerIndex].Deratives = new float[Mlp->Layers[LayerIndex].NumberOfNodes];
	
		BiasDeratives[LayerIndex].NumberOfNodes = Mlp->Layers[LayerIndex].NumberOfNodes;
		BiasDeratives[LayerIndex].Deratives = new float[Mlp->Layers[LayerIndex].NumberOfNodes];

	}
	for (int MatrixIndex = 0; MatrixIndex < Mlp->NumberOfLayers - 1; MatrixIndex++) {
		WeightDeratives[MatrixIndex].NumberOfColumns = Mlp->Matrices[MatrixIndex].NumberOfColumns;
		WeightDeratives[MatrixIndex].NumberOfRows = Mlp->Matrices[MatrixIndex].NumberOfRows;
		WeightDeratives[MatrixIndex].Weights =
			new float[Mlp->Matrices[MatrixIndex].NumberOfRows * Mlp->Matrices[MatrixIndex].NumberOfColumns];
		
	}

	for (int ExampleIndex = 0; ExampleIndex < NumberOfExamples; ExampleIndex++) {
		for (int NodeIndex = 0; NodeIndex < InputLayers[ExampleIndex].NumberOfNodes; NodeIndex++) {
			Mlp->Layers[0].Nodes[NodeIndex].Value = InputLayers[ExampleIndex].Nodes[NodeIndex].Value;
		}
		FarwardPass(Mlp);
	
		for (int NodeIndex = 0; NodeIndex < ExpectedOutputs[ExampleIndex].NumberOfNodes; NodeIndex++) {
			if (Mlp->Layers[Mlp->NumberOfLayers - 1].ActivationFunction != softmax) {
				LayerDeratives[Mlp->NumberOfLayers - 1].Deratives[NodeIndex] = 1.0f/2.0f * 
					(Mlp->Layers[Mlp->NumberOfLayers - 1].Nodes[NodeIndex].Value - ExpectedOutputs[ExampleIndex].Nodes[NodeIndex].Value);
			}
			else {
				LayerDeratives[Mlp->NumberOfLayers - 1].Deratives[NodeIndex]
					= Mlp->Layers[Mlp->NumberOfLayers - 1].Nodes[NodeIndex].Value - ExpectedOutputs[ExampleIndex].Nodes[NodeIndex].Value;
			}
		}
		for (int DerativeMatrixIndex = Mlp->NumberOfLayers - 2; DerativeMatrixIndex >= 0; DerativeMatrixIndex--) {
			for (int WeightRowIndex = 0; WeightRowIndex < WeightDeratives[DerativeMatrixIndex].NumberOfRows; 
				WeightRowIndex++) {
				for (int WeightColIndex = 0; WeightColIndex < WeightDeratives[DerativeMatrixIndex].NumberOfColumns;
					WeightColIndex++) {
					float ActivationFunctionDerivative = ComputeDerivativeOfActivationFunction(
									Mlp->Layers[DerativeMatrixIndex + 1].Nodes[WeightRowIndex].Value, Mlp->Layers[DerativeMatrixIndex + 1].ActivationFunction);
					

					if (ExampleIndex == 0) {
						WeightDeratives[DerativeMatrixIndex].Weights[WeightRowIndex *
							WeightDeratives[DerativeMatrixIndex].NumberOfColumns + WeightColIndex] =
								Mlp->Layers[DerativeMatrixIndex].Nodes[WeightColIndex].Value * ActivationFunctionDerivative * 
								LayerDeratives[DerativeMatrixIndex + 1].Deratives[WeightRowIndex];
					}
					else {
						WeightDeratives[DerativeMatrixIndex].Weights[WeightRowIndex *
							WeightDeratives[DerativeMatrixIndex].NumberOfColumns + WeightColIndex] +=
							Mlp->Layers[DerativeMatrixIndex].Nodes[WeightColIndex].Value * ActivationFunctionDerivative *
							LayerDeratives[DerativeMatrixIndex + 1].Deratives[WeightRowIndex];
					}
				}

			}

			for (int BiasNodeIndex = 0; BiasNodeIndex < Mlp->Layers[DerativeMatrixIndex + 1].NumberOfNodes; BiasNodeIndex++) {
				float ActivationFunctionDerivative = ComputeDerivativeOfActivationFunction(
					Mlp->Layers[DerativeMatrixIndex + 1].Nodes[BiasNodeIndex].Value, Mlp->Layers[DerativeMatrixIndex + 1].ActivationFunction);

				
				BiasDeratives[DerativeMatrixIndex + 1].Deratives[BiasNodeIndex] = ActivationFunctionDerivative *
					LayerDeratives[DerativeMatrixIndex + 1].Deratives[BiasNodeIndex];
			}

			layer PreviousLayer = Mlp->Layers[DerativeMatrixIndex];
			for (int PreviousLayerNodeIndex = 0; PreviousLayerNodeIndex < PreviousLayer.NumberOfNodes; PreviousLayerNodeIndex++) {
				layer CurrentLayer = Mlp->Layers[DerativeMatrixIndex + 1];
				float TotalDerative = 0;
				for (int CurrentLayerIndex = 0; CurrentLayerIndex < CurrentLayer.NumberOfNodes; CurrentLayerIndex++) {
					int NumberOfColumns = Mlp->Matrices[DerativeMatrixIndex].NumberOfColumns;
				
					float ActivationFunctionDerivative = ComputeDerivativeOfActivationFunction(
						Mlp->Layers[DerativeMatrixIndex + 1].Nodes[CurrentLayerIndex].Value, Mlp->Layers[DerativeMatrixIndex + 1].ActivationFunction);

	
					TotalDerative += Mlp->Matrices[DerativeMatrixIndex].Weights[CurrentLayerIndex * NumberOfColumns + PreviousLayerNodeIndex]
						* ActivationFunctionDerivative * LayerDeratives[DerativeMatrixIndex + 1].Deratives[CurrentLayerIndex];
				}
				LayerDeratives[DerativeMatrixIndex].Deratives[PreviousLayerNodeIndex] = TotalDerative;
			}

		
		}

	}

	for (int MatrixIndex = 0; MatrixIndex < Mlp->NumberOfLayers - 1; MatrixIndex++) {
		int NumberOfColumns = Mlp->Matrices[MatrixIndex].NumberOfColumns;
		int NumberOfRows = Mlp->Matrices[MatrixIndex].NumberOfRows;
		for (int WeightIndex = 0; WeightIndex < NumberOfColumns * NumberOfRows; WeightIndex++) {
			Mlp->Matrices[MatrixIndex].Weights[WeightIndex] -= WeightDeratives[MatrixIndex].Weights[WeightIndex] * 0.05;
		}
	}
	for (int LayerIndex = 1; LayerIndex < Mlp->NumberOfLayers; LayerIndex++) {
		for (int NodeIndex = 0; NodeIndex < BiasDeratives[LayerIndex].NumberOfNodes; NodeIndex++) {
			Mlp->Layers[LayerIndex].Nodes[NodeIndex].Bias -= BiasDeratives[LayerIndex].Deratives[NodeIndex] * 0.05;
		}
	}
}

void main() {
	srand(time(0));
	mlp Mlp;
	InitializeMLP(&Mlp, 2, 8, 2, 2);
	//LoadH5ToMlp(&Mlp, "W:\\dev\\mlp_scratch\\xor_model_keras(2).h5");
	RandomizeMlp(&Mlp);

	layer* Input = new layer[4];
	
	Input[0].Nodes = new node[2];
	Input[0].NumberOfNodes = 2;
	Input[0].Nodes[0].Value = 0;
	Input[0].Nodes[1].Value = 0;

	Input[1].Nodes = new node[2];
	Input[1].NumberOfNodes = 2;
	Input[1].Nodes[0].Value = 1;
	Input[1].Nodes[1].Value = 0;

	Input[2].Nodes = new node[2];
	Input[2].NumberOfNodes = 2;
	Input[2].Nodes[0].Value = 0;
	Input[2].Nodes[1].Value = 1;

	Input[3].Nodes = new node[2];
	Input[3].NumberOfNodes = 2;
	Input[3].Nodes[0].Value = 1;
	Input[3].Nodes[1].Value = 1;

	layer* ExpectedOutput = new layer[4];
	ExpectedOutput[0].Nodes = new node[2];
	ExpectedOutput[0].NumberOfNodes = 2;
	ExpectedOutput[0].Nodes[0].Value = 0;
	ExpectedOutput[0].Nodes[1].Value = 1;

	ExpectedOutput[1].Nodes = new node[2];
	ExpectedOutput[1].NumberOfNodes = 2;
	ExpectedOutput[1].Nodes[0].Value = 1;
	ExpectedOutput[1].Nodes[1].Value = 0;


	ExpectedOutput[2].Nodes = new node[2];
	ExpectedOutput[2].NumberOfNodes = 2;
	ExpectedOutput[2].Nodes[0].Value = 1;
	ExpectedOutput[2].Nodes[1].Value = 0;


	ExpectedOutput[3].Nodes = new node[2];
	ExpectedOutput[3].NumberOfNodes = 2;
	ExpectedOutput[3].Nodes[0].Value = 0;
	ExpectedOutput[3].Nodes[1].Value = 1;

	for(int Iteration = 0; Iteration < 50000; Iteration++)
	TrainMlp(&Mlp, Input, ExpectedOutput, 4);

	Mlp.Layers[0].Nodes[0].Value = 0;
	Mlp.Layers[0].Nodes[1].Value = 0;
	FarwardPass(&Mlp);
	int FinalLayer = Mlp.NumberOfLayers - 1;
	std::cout << Mlp.Layers[FinalLayer].Nodes[0].Value << std::endl;
	std::cout << Mlp.Layers[FinalLayer].Nodes[1].Value << "\n\n";


	Mlp.Layers[0].Nodes[0].Value = 0;
	Mlp.Layers[0].Nodes[1].Value = 1;
	FarwardPass(&Mlp);
	std::cout << Mlp.Layers[FinalLayer].Nodes[0].Value << std::endl;
	std::cout << Mlp.Layers[FinalLayer].Nodes[1].Value << "\n\n";

	Mlp.Layers[0].Nodes[0].Value = 1;
	Mlp.Layers[0].Nodes[1].Value = 0;
	FarwardPass(&Mlp);
	std::cout << Mlp.Layers[FinalLayer].Nodes[0].Value << std::endl;
	std::cout << Mlp.Layers[FinalLayer].Nodes[1].Value << "\n\n";

	Mlp.Layers[0].Nodes[0].Value = 1;
	Mlp.Layers[0].Nodes[1].Value = 1;
	FarwardPass(&Mlp);
	std::cout << Mlp.Layers[FinalLayer].Nodes[0].Value << std::endl;
	std::cout << Mlp.Layers[FinalLayer].Nodes[1].Value << "\n\n";

	//std::cout << "finished";
}