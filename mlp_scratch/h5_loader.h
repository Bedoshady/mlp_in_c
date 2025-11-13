#ifndef h5_loader
#define h5_loader 1


#include <iostream>
#include <vector>
#include <string>
#include "hdf5.h"

struct bias_array {
    float* Biases;
    int NumberOfBiases;
};

struct weight_matrix {
    float* Weights;
    int NumberOfColumns, NumberOfRows;
};

struct mlp_information {
    bias_array* LayerBiases;
    weight_matrix* MlpMatrices;
    int NumberOfLayers;
};
struct h5_dataset {
    float* Weights;
    int NumberOfColumns, NumberOfRows;
};

void TransposeDataset(h5_dataset* Input) {
    int NewNumberOfColumns = Input->NumberOfRows;
    int NewNumberOfRows = Input->NumberOfColumns;

    float* NewDataset = new float[NewNumberOfColumns * NewNumberOfRows];
    for (int RowIndex = 0; RowIndex < NewNumberOfRows; RowIndex++) {
        for (int ColIndex = 0; ColIndex < NewNumberOfColumns; ColIndex++) {
            NewDataset[RowIndex * NewNumberOfColumns + ColIndex] =
                Input->Weights[ColIndex * Input->NumberOfColumns + RowIndex];
        }
    }
    free(Input->Weights);
    Input->NumberOfRows = NewNumberOfRows;
    Input->NumberOfColumns = NewNumberOfColumns;
    Input->Weights = NewDataset;
}

h5_dataset LoadDatasetsValues(hid_t DatasetId) {
    h5_dataset Result = {};

    hid_t SpaceId = H5Dget_space(DatasetId);
    int rank = H5Sget_simple_extent_ndims(SpaceId);
    hsize_t Dimensions[2] = { 1, 1 };
    H5Sget_simple_extent_dims(SpaceId, Dimensions, nullptr);

    Result.NumberOfRows = Dimensions[0];
    Result.NumberOfColumns = Dimensions[1];

    hsize_t TotalSize = 1;
    for (int DimensionIndex = 0; DimensionIndex < 2; DimensionIndex++) {
        TotalSize *= Dimensions[DimensionIndex];
    }

    float* DatasetData = new float[TotalSize];
    if (H5Dread(DatasetId, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, DatasetData) >= 0) {
        Result.Weights = DatasetData;
    }
    else {
        std::cout << "Couldnt read dataset\n";
    }

    H5Sclose(SpaceId);
    return Result;
}
mlp_information LoadH5File(const char* FileName) {
    H5Eset_auto2(H5E_DEFAULT, NULL, NULL);
    mlp_information Result = {};
    H5open();

    hid_t FileId = H5Fopen(FileName, H5F_ACC_RDONLY, H5P_DEFAULT);
    if (FileId < 0) {
        std::cerr << "Failed to open file\n";
    }

    hid_t ModelWeightsGroupId = H5Gopen2(FileId, "model_weights", H5P_DEFAULT);
    hsize_t NumberOfObject;
    H5Gget_num_objs(ModelWeightsGroupId, &NumberOfObject);

    int NumberOfLayers = 0;
    for (int ObjectIndex = 0; ObjectIndex < NumberOfObject; ObjectIndex++) {
        char ObjectName[256];
        H5Gget_objname_by_idx(ModelWeightsGroupId, ObjectIndex, ObjectName, 256);
        const char* DenseName = "dense";
        bool IsDenseGroup = true;
        for (int CharIndex = 0; CharIndex < strlen(DenseName); CharIndex++) {
            if (ObjectName[CharIndex] != DenseName[CharIndex]) {
                IsDenseGroup = false;
                break;
            }
        }
        if (IsDenseGroup)
            NumberOfLayers++;
    }

    Result.NumberOfLayers = NumberOfLayers;
    Result.LayerBiases = new bias_array[NumberOfLayers];
    Result.MlpMatrices = new weight_matrix[NumberOfLayers];

    int LayerIndex = 0;
    for (int ObjectIndex = 0; ObjectIndex < NumberOfObject; ObjectIndex++) {
        char ObjectName[256];
        H5Gget_objname_by_idx(ModelWeightsGroupId, ObjectIndex, ObjectName, 256);
        const char* DenseName = "dense";
        bool IsDenseGroup = true;
        for (int CharIndex = 0; CharIndex < strlen(DenseName); CharIndex++) {
            if (ObjectName[CharIndex] != DenseName[CharIndex]) {
                IsDenseGroup = false;
                break;
            }
        }
        if (IsDenseGroup)
        {
            hid_t CurrentGroupId = H5Gopen2(ModelWeightsGroupId, ObjectName, H5P_DEFAULT);
            hsize_t NumberOfObjectsInGroup;
            H5Gget_num_objs(CurrentGroupId, &NumberOfObjectsInGroup);
            while (NumberOfObjectsInGroup != 2) {
                char NewName[256];
                H5Gget_objname_by_idx(CurrentGroupId, 0, NewName, 256);
                CurrentGroupId = H5Gopen2(CurrentGroupId, NewName, H5P_DEFAULT);
                H5Gget_num_objs(CurrentGroupId, &NumberOfObjectsInGroup);
            }

            hid_t DatasetWeightId = H5Dopen2(CurrentGroupId, "kernel:0", H5P_DEFAULT);
            if(DatasetWeightId < 0)
                DatasetWeightId = H5Dopen2(CurrentGroupId, "kernel", H5P_DEFAULT);
           
            hid_t DatasetBiasId = H5Dopen2(CurrentGroupId, "bias:0", H5P_DEFAULT);
            if (DatasetBiasId < 0)
                DatasetBiasId = H5Dopen2(CurrentGroupId, "bias", H5P_DEFAULT);

            h5_dataset Weights = LoadDatasetsValues(DatasetWeightId);
            TransposeDataset(&Weights);

            Result.MlpMatrices[LayerIndex].NumberOfRows = Weights.NumberOfRows;
            Result.MlpMatrices[LayerIndex].NumberOfColumns = Weights.NumberOfColumns;
            Result.MlpMatrices[LayerIndex].Weights = Weights.Weights;

            h5_dataset Biases = LoadDatasetsValues(DatasetBiasId);
            Result.LayerBiases[LayerIndex].NumberOfBiases = Biases.NumberOfRows;
            Result.LayerBiases[LayerIndex].Biases = Biases.Weights;


            H5Gclose(DatasetWeightId);
            H5Gclose(DatasetBiasId);

            //H5Gclose(DatasetGroupId);
            //H5Gclose(DenseGroupId);

            LayerIndex++;
        }
    }

    H5Gclose(ModelWeightsGroupId);
    H5close();
    return Result;
}

#endif
