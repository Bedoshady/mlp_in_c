#ifndef mnist_loader
#define mnist_loader 1

#include<stdio.h>

struct training_example{
	float ExpectedOutput;
	float* InputValues;
	int NumberOfInputs;
};

struct training_examples_array {
	training_example* Examples;
	int NumberOfExamples;
};

void LoadExamplesFromCSV(const char* FileName) {
	FILE* File = fopen(FileName, "r");
	if (File) {
		fseek(File, 0L, SEEK_END);
		int FileSize = ftell(File);
		fseek(File, 0, SEEK_SET);

		char* FileBuffer = (char*)malloc(FileSize);

		for (int CharacterIndex = 0; CharacterIndex < FileSize; CharacterIndex++) {}



		fclose(File);
	}
	else {
		std::cout << "Couldnt open CSV\n";
	}
}

#endif // !mnist_loader
