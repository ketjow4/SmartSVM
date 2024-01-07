// #pragma once

// #include <filesystem>
// #include "libDataset/Dataset.h"


// #include <pybind11/embed.h>
// #include <pybind11/numpy.h>


// namespace py = pybind11;
// using namespace py::literals;

// class ExtraTreeWrapper
// {
// public:
// 	ExtraTreeWrapper();

// 	void load(std::filesystem::path path);

// 	void save(std::filesystem::path path);

// 	void train(const dataset::Dataset<std::vector<float>, float>& dataset);

// 	float predict(const gsl::span<const float> sample) const;

// 	std::vector<float> predict(const dataset::Dataset<std::vector<float>, float>& dataset);

// private:
// 	py::array_t<float> to_matrix(const dataset::Dataset<std::vector<float>, float>& dataset);

// 	py::object m_pickle;
// 	py::object m_extraTree;
// 	py::object m_io;
// 	py::object m_clf;
// };