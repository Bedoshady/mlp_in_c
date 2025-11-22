#include"mnist_loader.h"
#include<iostream>
#include<math.h>
//#include"h5_loader.h"
#include <chrono>

struct weight_matrix {
	float* Weights;
	int NumberOfColumns, NumberOfRows;
};


struct weight_matrix_derivative {
	float* Weights;
	int NumberOfColumns, NumberOfRows;
};


enum activation_function {
	sigmoid,
	relu,
	softmax
};

struct node {
	float Value;
	float Bias;
//	float IntermediateValue;
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
		
		if (LayerIndex == 1) {
			Mlp->Layers[LayerIndex].NumberOfNodes = 128;
			Mlp->Layers[LayerIndex].Nodes = new node[128];
		}
		else if(LayerIndex == 2){
			Mlp->Layers[LayerIndex].NumberOfNodes = 64;
			Mlp->Layers[LayerIndex].Nodes = new node[64];
		}
		else {
			Mlp->Layers[LayerIndex].NumberOfNodes = 50;
			Mlp->Layers[LayerIndex].Nodes = new node[50];

		}
		
		
		
		/*
		Mlp->Layers[LayerIndex].NumberOfNodes = NodesPerHiddenLayer;
		Mlp->Layers[LayerIndex].Nodes = new node[Mlp->Layers[LayerIndex].NumberOfNodes];
		*/
		
		
		Mlp->Layers[LayerIndex].ActivationFunction = relu;
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
			Mlp->Matrices[LayerIndex].Weights[WeightIndex] =  ((float)rand() / RAND_MAX) * 2.0f * (1.0f / sqrt((float)Mlp->Layers[LayerIndex].NumberOfNodes))
					+ -(1.0f / sqrt((float)Mlp->Layers[LayerIndex].NumberOfNodes));
		}
	
	}

}



inline float ApplyActivationFunction(float Input, activation_function Function) {
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
	

	//return (Function == sigmoid) * 1.0f / (1.0f + exp(-Input)) + (Function == relu) * (Input > 0) * Input;
	//std::cout << "Unknown activation function";
	//return 1000;

}

void MatrixMultiply(weight_matrix* Matrix, layer* Input, layer* Output, bool IsOutputLayer) {
	float Total = 0;
	float Max = -1;
	const int NumberOfRows = Matrix->NumberOfRows;
	const int NumberOfColumns = Matrix->NumberOfColumns;
	int WeightIndex = 0;
	for (int RowIndex = 0; RowIndex < NumberOfRows; RowIndex++) {
		Output->Nodes[RowIndex].Value = 0;
		for (int ColIndex = 0; ColIndex < NumberOfColumns; ColIndex++) {
			Output->Nodes[RowIndex].Value += Input->Nodes[ColIndex].Value *  Matrix->Weights[WeightIndex];
			WeightIndex++;
			
		}
		Output->Nodes[RowIndex].Value += Output->Nodes[RowIndex].Bias;
		//Output->Nodes[RowIndex].IntermediateValue = Output->Nodes[RowIndex].Value;

		
	}
	Max = Output->Nodes[0].Value;
	for (int RowIndex = 0; RowIndex < NumberOfRows; RowIndex++) {
		if (!IsOutputLayer || Output->ActivationFunction != softmax)
			Output->Nodes[RowIndex].Value = ApplyActivationFunction(Output->Nodes[RowIndex].Value, Output->ActivationFunction);
		if (Output->Nodes[RowIndex].Value > Max) {
			Max = Output->Nodes[RowIndex].Value;
		}
	}

	if (IsOutputLayer && Output->ActivationFunction == softmax) {
		Total = 0;
		for (int NodeIndex = 0; NodeIndex < Output->NumberOfNodes; NodeIndex++) {
			Output->Nodes[NodeIndex].Value = exp(Output->Nodes[NodeIndex].Value - Max);
			Total += Output->Nodes[NodeIndex].Value;
		}
		
		for (int NodeIndex = 0; NodeIndex < Output->NumberOfNodes; NodeIndex++) {
			Output->Nodes[NodeIndex].Value /= Total;
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
*/

struct layer_deratives {
	float* Deratives;
	int NumberOfNodes;
};

inline float ComputeDerivativeOfActivationFunction(float InputValue, activation_function Function) {
	switch (Function)
	{
	case sigmoid: {
		return InputValue * (1 - InputValue);
	}break;
	case relu: {
		return InputValue > 0;
	}break;
	case softmax: {
		return 1;
	}break;
	}
	
	//return (Function == sigmoid) * InputValue * (1 - InputValue) + (Function == sigmoid) * (InputValue > 0) + (Function == softmax);
	std::cout << "Unknown activation function";
	return -1000;
}

void TrainMlp(mlp* Mlp, layer* InputLayers, layer* ExpectedOutputs, int NumberOfExamples) {
	const float LearningRate = 0.01;
	
	layer_deratives* LayerDeratives = new layer_deratives[Mlp->NumberOfLayers - 1];
	weight_matrix_derivative* WeightDeratives = new weight_matrix_derivative[Mlp->NumberOfLayers - 1];
	
	layer_deratives* BiasDeratives = new layer_deratives[Mlp->NumberOfLayers - 1];

	for (int LayerIndex = 0; LayerIndex < Mlp->NumberOfLayers - 1; LayerIndex++) {
		LayerDeratives[LayerIndex].NumberOfNodes = Mlp->Layers[LayerIndex + 1].NumberOfNodes;
		LayerDeratives[LayerIndex].Deratives = new float[Mlp->Layers[LayerIndex + 1].NumberOfNodes];
	
		BiasDeratives[LayerIndex].NumberOfNodes = Mlp->Layers[LayerIndex + 1].NumberOfNodes;
		BiasDeratives[LayerIndex].Deratives = new float[Mlp->Layers[LayerIndex + 1].NumberOfNodes]{};

	}
	for (int MatrixIndex = 0; MatrixIndex < Mlp->NumberOfLayers - 1; MatrixIndex++) {
		WeightDeratives[MatrixIndex].NumberOfColumns = Mlp->Matrices[MatrixIndex].NumberOfColumns;
		WeightDeratives[MatrixIndex].NumberOfRows = Mlp->Matrices[MatrixIndex].NumberOfRows;
		WeightDeratives[MatrixIndex].Weights = new float[Mlp->Matrices[MatrixIndex].NumberOfRows * Mlp->Matrices[MatrixIndex].NumberOfColumns]{};
		
	}
	for (int ExampleIndex = 0; ExampleIndex < NumberOfExamples; ExampleIndex++) {

		for (int NodeIndex = 0; NodeIndex < InputLayers[ExampleIndex].NumberOfNodes; NodeIndex++) {
			Mlp->Layers[0].Nodes[NodeIndex].Value = InputLayers[ExampleIndex].Nodes[NodeIndex].Value;
		}
		FarwardPass(Mlp);
		/*
		std::chrono::steady_clock::time_point Farward = std::chrono::steady_clock::now();
		if (ExampleIndex % 1000 == 0)
		std::cout << "Farward pass time = " << std::chrono::duration_cast<std::chrono::microseconds>(Farward - begin).count() << "[us]" << std::endl;
		*/
		
		for (int NodeIndex = 0; NodeIndex < ExpectedOutputs[ExampleIndex].NumberOfNodes; NodeIndex++) {
			if (Mlp->Layers[Mlp->NumberOfLayers - 2].ActivationFunction != softmax) {
				LayerDeratives[Mlp->NumberOfLayers - 2].Deratives[NodeIndex] = 1.0f/2.0f * (Mlp->Layers[Mlp->NumberOfLayers - 1].Nodes[NodeIndex].Value - ExpectedOutputs[ExampleIndex].Nodes[NodeIndex].Value);
			}
			else {
				LayerDeratives[Mlp->NumberOfLayers - 2].Deratives[NodeIndex] = Mlp->Layers[Mlp->NumberOfLayers - 1].Nodes[NodeIndex].Value - ExpectedOutputs[ExampleIndex].Nodes[NodeIndex].Value;
			}
		}
		
		//std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

		for (int DerativeMatrixIndex = Mlp->NumberOfLayers - 2; DerativeMatrixIndex >= 0; DerativeMatrixIndex--) {
			int NumberOfRows = WeightDeratives[DerativeMatrixIndex].NumberOfRows;
			int NumberOfColumns = WeightDeratives[DerativeMatrixIndex].NumberOfColumns;
			activation_function ActiveFunction = Mlp->Layers[DerativeMatrixIndex + 1].ActivationFunction;
			
			layer* CurrentLayer = &Mlp->Layers[DerativeMatrixIndex + 1];
			layer* PreviousLayer = &Mlp->Layers[DerativeMatrixIndex];
			weight_matrix_derivative* CurrentDerivativeMatrix = &WeightDeratives[DerativeMatrixIndex];
			layer_deratives* CurrentLayerDerative = &LayerDeratives[DerativeMatrixIndex - 1];
			layer_deratives* CurrentBiasDerative = &BiasDeratives[DerativeMatrixIndex];
			
			for (int WeightRowIndex = 0; WeightRowIndex < NumberOfRows; WeightRowIndex++) {
				float ActivationFunctionDerivative = (ActiveFunction == relu) ? (CurrentLayer->Nodes[WeightRowIndex].Value > 0) : (ActiveFunction == softmax);// 
				//ComputeDerivativeOfActivationFunction(CurrentLayer->Nodes[WeightRowIndex].Value, ActiveFunction);
				float CommonDelta = ActivationFunctionDerivative * LayerDeratives[DerativeMatrixIndex].Deratives[WeightRowIndex];

				for (int WeightColIndex = 0; WeightColIndex < NumberOfColumns; WeightColIndex++) {
					CurrentDerivativeMatrix->Weights[WeightRowIndex * NumberOfColumns + WeightColIndex] = PreviousLayer->Nodes[WeightColIndex].Value * CommonDelta;
					
				}
				if (DerativeMatrixIndex != 0) {
					for (int WeightColIndex = 0; WeightColIndex < NumberOfColumns; WeightColIndex++) {
						CurrentLayerDerative->Deratives[WeightColIndex] += Mlp->Matrices[DerativeMatrixIndex].Weights[WeightRowIndex * NumberOfColumns + WeightColIndex] * CommonDelta;
					}
				}
				CurrentBiasDerative->Deratives[WeightRowIndex] = CommonDelta;
			}
		
		}
		//std::chrono::steady_clock::time_point Backprob = std::chrono::steady_clock::now();
		//std::cout << "Weight matrix update time = " << std::chrono::duration_cast<std::chrono::microseconds>(Backprob - begin).count() << "[us]" << std::endl;

		//if (ExampleIndex % 100 == 0) {
		for (int MatrixIndex = 0; MatrixIndex < Mlp->NumberOfLayers - 1; MatrixIndex++) {
			int NumberOfColumns = Mlp->Matrices[MatrixIndex].NumberOfColumns;
			int NumberOfRows = Mlp->Matrices[MatrixIndex].NumberOfRows;
			int WeightIndex = 0;
			int LayerIndex = MatrixIndex;

			layer* CurrentLayer = &Mlp->Layers[LayerIndex + 1];
			weight_matrix_derivative* CurrentDerivativeMatrix = &WeightDeratives[MatrixIndex];
			layer_deratives* CurrentLayerDerative = &LayerDeratives[LayerIndex];
			layer_deratives* CurrentBiasDerative = &BiasDeratives[LayerIndex];
			weight_matrix* CurrentWeightMatrix = &Mlp->Matrices[MatrixIndex];

			for (int WeightRowIndex = 0; WeightRowIndex < NumberOfRows; WeightRowIndex++) {
				for (int WeightColIndex = 0; WeightColIndex < NumberOfColumns; WeightColIndex++) {
					CurrentWeightMatrix->Weights[WeightIndex] -= CurrentDerivativeMatrix->Weights[WeightIndex] * LearningRate;
					//CurrentDerivativeMatrix->Weights[WeightIndex] = 0;
					WeightIndex++;
				}	
				CurrentLayer->Nodes[WeightRowIndex].Bias -= CurrentBiasDerative->Deratives[WeightRowIndex] * LearningRate;
				CurrentLayerDerative->Deratives[WeightRowIndex] = 0;
				//CurrentBiasDerative->Deratives[WeightRowIndex] = 0;

			}
		}
		

	}
	
	for (int LayerIndex = 0; LayerIndex < Mlp->NumberOfLayers - 1; LayerIndex++) {
		delete[] LayerDeratives[LayerIndex].Deratives;
		delete[] BiasDeratives[LayerIndex].Deratives;
		delete[] WeightDeratives[LayerIndex].Weights;
	}
	delete[] LayerDeratives;
	delete[] BiasDeratives;
	delete[] WeightDeratives;


}

void main() {
	srand(time(0));
	mlp Mlp;
	InitializeMLP(&Mlp, 784, 32, 2, 10);
	training_examples_array TrainingData = LoadExamplesFromCSV("W:\\dev\\mlp_scratch\\mnist_train.csv");
	//LoadH5ToMlp(&Mlp, "W:\\dev\\mlp_scratch\\xor_model_keras(2).h5");
	RandomizeMlp(&Mlp);

	int NumberOfExamples = 60000;
	layer *Input = new layer[NumberOfExamples];
	layer* ExpectedOutput = new layer[NumberOfExamples];


	for (int TrainingExample = 0; TrainingExample < NumberOfExamples; TrainingExample++) {
		Input[TrainingExample].NumberOfNodes = 784;
		Input[TrainingExample].Nodes = new node[784];

		for (int i = 0; i < 784; i++) {
			/*
			if (TrainingData.Examples[TrainingExample].InputValues[i] > 128) {
				Input[TrainingExample].Nodes[i].Value = 1;
			}
			else {
				Input[TrainingExample].Nodes[i].Value = 0;
			}
			*/
			Input[TrainingExample].Nodes[i].Value = TrainingData.Examples[TrainingExample].InputValues[i] / 255.0f;
			if (TrainingData.Examples[TrainingExample].InputValues[i] > 255) {
				std::cout << "Error in reading data\n";
			}
		}

		ExpectedOutput[TrainingExample].NumberOfNodes = 10;
		ExpectedOutput[TrainingExample].Nodes = new node[10];

		for (int i = 0; i < 10; i++) {
			ExpectedOutput[TrainingExample].Nodes[i].Value = 0;
		}
		ExpectedOutput[TrainingExample].Nodes[(int)(TrainingData.Examples[TrainingExample].ExpectedOutput)].Value = 1;
		delete[] TrainingData.Examples[TrainingExample].InputValues;

	}
	delete[] TrainingData.Examples;

	for (int Iteration = 0; Iteration < 2; Iteration++)
	{
		std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
		std::cout << "Iteration Number: " << Iteration << "\n";
		TrainMlp(&Mlp, Input, ExpectedOutput, NumberOfExamples);
		std::chrono::steady_clock::time_point Backprob = std::chrono::steady_clock::now();
		std::cout << "Backprob time = " << std::chrono::duration_cast<std::chrono::seconds>(Backprob - begin).count() << "[us]" << std::endl;

	}	

	training_examples_array TestData = LoadExamplesFromCSV("W:\\dev\\mlp_scratch\\mnist_test.csv");
	int NumberOfCorrect = 0;
	for (int TestExample = 0; TestExample < TestData.NumberOfExamples; TestExample++) {
		
		for (int i = 0; i < 784; i++) {
			if (TestData.Examples[TestExample].InputValues[i] > 128) {
				Mlp.Layers[0].Nodes[i].Value = 1;
			}
			else {
				Mlp.Layers[0].Nodes[i].Value = 0;
			}
			//Mlp.Layers[0].Nodes[i].Value = TestData.Examples[TestExample].InputValues[i] / 255.0f;
		}
		FarwardPass(&Mlp);
		int OutputIndex = 0;
		float Probability = Mlp.Layers[Mlp.NumberOfLayers - 1].Nodes[0].Value;
		for (int i = 1; i < 10; i++) {
			if (Mlp.Layers[Mlp.NumberOfLayers - 1].Nodes[i].Value > Probability)
			{
				OutputIndex = i;
				Probability = Mlp.Layers[Mlp.NumberOfLayers - 1].Nodes[i].Value;
			}
		}
		/*
		
		*/

		if (OutputIndex == TestData.Examples[TestExample].ExpectedOutput)
			NumberOfCorrect++;
		else {
			/*
			std::cout << "Incorrect Expected: " << TestData.Examples[TestExample].ExpectedOutput <<
				" output is: " << OutputIndex << "\n";
			for (int i = 0; i < 10; i++) {
				std::cout << "Probability for " << i << " " << Mlp.Layers[Mlp.NumberOfLayers - 1].Nodes[i].Value << "\n";
			}
			*/
		}
	}
	std::cout << "Number of correct is :" << NumberOfCorrect << "\n";
	//std::cout << "finished";
}