
// #include "ExtraTreeWrapper.h"

// ExtraTreeWrapper::ExtraTreeWrapper()
// {
// 	m_pickle = py::module_::import("pickle");
// 	m_extraTree = py::module_::import("sklearn.ensemble");
// 	m_io = py::module_::import("io");
// }

// void ExtraTreeWrapper::load(std::filesystem::path path)
// {
// 	//auto file = m_io.attr("open")(R"(D:\ENSEMBLE_TEST_BED2 - Copy\2D_shapes\scikit_20210921-142343\ExtraTree_0_20210921-142357__0.pkl)", "rb");
// 	auto file = m_io.attr("open")(path.string(), "rb");
// 	m_clf = m_pickle.attr("load")(file);
// }

// void ExtraTreeWrapper::save(std::filesystem::path path)
// {
// 	auto file = m_io.attr("open")(path.string(), "wb");
// 	auto protocol_version = 4;
// 	m_clf = m_pickle.attr("dump")(m_clf, file, protocol_version);
// }

// void ExtraTreeWrapper::train(const dataset::Dataset<std::vector<float>, float>& dataset)
// {
// 	m_clf = m_extraTree.attr("ExtraTreesClassifier")("n_estimators"_a = 10, "random_state"_a = 42);

// 	auto trainSet = to_matrix(dataset);
// 	auto l = dataset.getLabels();
// 	py::array labels({static_cast<int>(dataset.getLabels().size())}, l.data());

// 	m_clf.attr("fit")(trainSet, labels);
// }

// float ExtraTreeWrapper::predict(const gsl::span<const float> sample) const
// {
// 	py::array pyVect({1, static_cast<int>(sample.size())}, sample.data());
// 	auto response = m_clf.attr("predict")(pyVect);

// 	auto transformed = response.cast<py::array_t<float>>();
// 	return transformed.at(0);
// }

// std::vector<float> ExtraTreeWrapper::predict(const dataset::Dataset<std::vector<float>, float>& dataset)
// {
// 	auto samplesSize = dataset.getSamples().size();

// 	std::vector<float> responses;
// 	responses.reserve(samplesSize);

// 	auto mat = to_matrix(dataset);
// 	auto response = m_clf.attr("predict")(mat);
// 	auto transformed = response.cast<py::array_t<float>>();

// 	for (auto i = 0u; i < samplesSize; ++i)
// 	{
// 		auto converted = transformed.at(i);
// 		responses.emplace_back(converted);
// 	}

// 	return responses;
// }

// py::array_t<float> ExtraTreeWrapper::to_matrix(const dataset::Dataset<std::vector<float>, float>& dataset)
// {
// 	auto samples = dataset.getSamples();

// 	size_t N = samples.size();
// 	size_t M = samples[0].size();

// 	py::array_t<float, py::array::c_style> arr({N, M});

// 	auto ra = arr.mutable_unchecked();

// 	for (size_t i = 0; i < N; i++)
// 	{
// 		for (size_t j = 0; j < M; j++)
// 		{
// 			ra(i, j) = samples[i][j];
// 		}
// 	}
// 	return arr;
// }
