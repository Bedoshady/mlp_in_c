#ifndef mnist_loader
#define mnist_loader 1


#define _CRT_SECURE_NO_WARNINGS 1
#include<stdio.h>
#include<stdlib.h>


struct training_example{
	int ExpectedOutput;
	int* InputValues;
	int NumberOfInputs;
};

struct training_examples_array {
	training_example* Examples;
	int NumberOfExamples;
};


int ExtractIntFromBuffer(char* FileBuffer, int WordStartIndex, int WordEndIndex) {
	int CurrentInt;
	char* ExtractedIntString = new char[WordEndIndex - WordStartIndex + 2];
	for (int i = 0; i < WordEndIndex - WordStartIndex + 1; i++) {
		ExtractedIntString[i] = FileBuffer[WordStartIndex + i];
	}
	ExtractedIntString[WordEndIndex - WordStartIndex + 1] = '\n';
	CurrentInt = atoi(ExtractedIntString);
	free(ExtractedIntString);
	return CurrentInt;
}

training_examples_array LoadExamplesFromCSV(const char* FileName) {
	FILE* File = fopen(FileName, "rb");
	if (File) {
		fseek(File, 0L, SEEK_END);
		int FileSize = ftell(File);
		fseek(File, 0, SEEK_SET);

		char* FileBuffer = (char*)malloc(FileSize);
		fread(FileBuffer, 1, FileSize, File);

		int NumberOfRows = 0;
		for (int CharacterIndex = 0; CharacterIndex < FileSize - 1; CharacterIndex++) {
			if (FileBuffer[CharacterIndex] == '\r' && FileBuffer[CharacterIndex + 1] == '\n') {
				NumberOfRows++;
				CharacterIndex++;
				if(NumberOfRows == 1)
				printf("%d\n", CharacterIndex);
			}
		}

		NumberOfRows++;
		training_examples_array FinalArray;
		FinalArray.NumberOfExamples = NumberOfRows;
		FinalArray.Examples = new training_example[NumberOfRows];

		for (int TrainingExample = 0; TrainingExample < NumberOfRows; TrainingExample++) {
			FinalArray.Examples[TrainingExample].NumberOfInputs = 784;
			FinalArray.Examples[TrainingExample].InputValues = new int[784];
		}

		int TrainingExample = 0;
		int InputIndex = 0;
		
		int WordStartIndex = 0;
		int WordEndIndex = -1;

		for (int CharacterIndex = 0; CharacterIndex < FileSize; CharacterIndex++) {
			if (FileBuffer[CharacterIndex] == ',') {
				int CurrentInt = ExtractIntFromBuffer(FileBuffer, WordStartIndex, WordEndIndex);
				if (InputIndex == 0) {
					FinalArray.Examples[TrainingExample].ExpectedOutput = CurrentInt;
				}
				else {
					FinalArray.Examples[TrainingExample].InputValues[InputIndex - 1] = CurrentInt;
				}

				WordStartIndex = CharacterIndex + 1;
				WordEndIndex = CharacterIndex;

				InputIndex++;
			}
			else if (FileBuffer[CharacterIndex] == '\n') {
				int CurrentInt = ExtractIntFromBuffer(FileBuffer, WordStartIndex, WordEndIndex);
				FinalArray.Examples[TrainingExample].InputValues[InputIndex - 1] = CurrentInt;
				
				WordStartIndex = CharacterIndex + 1;
				WordEndIndex = CharacterIndex;

				InputIndex = 0;
				TrainingExample++;
			}
			else {
				WordEndIndex++;
			}
		}

		fclose(File);
		return FinalArray;
	}
	else {
		printf("Couldnt open CSV\n");
	}
	return training_examples_array();
}

#endif // !mnist_loader
