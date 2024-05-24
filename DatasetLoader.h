#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "libGeneticSvm/IDatasetLoader.h"

namespace py = pybind11;

class NumpyDatasetLoader : public genetic::IDatasetLoader
{
public:
    NumpyDatasetLoader() = default;

    NumpyDatasetLoader(py::array_t<float> tr_x, py::array_t<float> tr_y,
                       py::array_t<float> val_x, py::array_t<float> val_y,
                       py::array_t<float> test_x, py::array_t<float> test_y);


    const dataset::Dataset<std::vector<float>, float>& getTraningSet() override { return m_traningSet; }
    const dataset::Dataset<std::vector<float>, float>& getValidationSet() override { return m_validationSet; }
    const dataset::Dataset<std::vector<float>, float>& getTestSet() override { return m_testSet; }
    bool isDataLoaded() const override { return true; }
    const std::vector<float>& scalingVectorMin() override { throw std::runtime_error("Not implemented"); }
    const std::vector<float>& scalingVectorMax() override { throw std::runtime_error("Not implemented"); }

    ~NumpyDatasetLoader() override {}

private:
    dataset::Dataset<std::vector<float>, float> m_traningSet;
    dataset::Dataset<std::vector<float>, float> m_validationSet;
    dataset::Dataset<std::vector<float>, float> m_testSet;
};

dataset::Dataset<std::vector<float>, float> convertToDataset(py::array_t<float> data, py::array_t<float> labels);

std::vector<float> numpyToVector(py::array_t<float> inputArray);

std::vector<std::vector<float>> numpyToVectorOfVectors(py::array_t<float> inputArray);

