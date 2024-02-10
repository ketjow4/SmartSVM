
#include "DatasetLoader.h"
#include <libDataset/Dataset.h>

#include <libDataset/CsvReader.h>

std::vector<std::vector<float>> numpyToVectorOfVectors(py::array_t<float> inputArray)
{
    py::buffer_info bufInfo = inputArray.request();

    if (bufInfo.ndim != 2) {
        throw std::runtime_error("Input array must be 2-dimensional.");
    }

    std::vector<std::vector<float>> result;

    auto ptr = static_cast<float*>(bufInfo.ptr);
    for (int i = 0; i < bufInfo.shape[0]; ++i) {
        std::vector<float> row(ptr + i * bufInfo.shape[1], ptr + (i + 1) * bufInfo.shape[1]);
        result.push_back(row);
    }

    return result;
}


std::vector<float> numpyToVector(py::array_t<float> inputArray)
{
    py::buffer_info bufInfo = inputArray.request();

    if (bufInfo.ndim != 1) {
        throw std::runtime_error("Input array must be 1-dimensional.");
    }

    float* ptr = static_cast<float*>(bufInfo.ptr);
    std::vector<float> vec;
    vec.reserve(bufInfo.size);
    std::copy(ptr, ptr + bufInfo.size, vec.data());

    return vec;
}


dataset::Dataset<std::vector<float>, float> convertToDataset(py::array_t<float> data, py::array_t<float> labels)
{
    auto samples = numpyToVectorOfVectors(data);
    auto labelsConverted = numpyToVector(labels);

    dataset::Dataset<std::vector<float>, float> dataset;

    for (auto i = 0u; i < samples.size(); i++)
    {
        dataset.addSample(samples[i], labelsConverted[i]);
    }

    return dataset;
}





NumpyDatasetLoader::NumpyDatasetLoader(py::array_t<float> tr_x, py::array_t<float> tr_y,
                                       py::array_t<float> val_x, py::array_t<float> val_y, py::array_t<float> test_x,
                                       py::array_t<float> test_y)
	: m_traningSet(convertToDataset(tr_x, tr_y))
	  , m_validationSet(convertToDataset(val_x, val_y))
	  , m_testSet(convertToDataset(test_x, test_y))
{
}
