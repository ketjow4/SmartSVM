
#include "libSvmImplementation.h"

#include <fstream>
#include <iostream>
#include <map>
#include <set>

#include "libPlatform/loguru.hpp"
//#include "libSvmComponents/IGroupPropagation.h"

namespace phd { namespace svm
{
double vectorDistance(svm_node* first, svm_node* second);

libSvmImplementation::~libSvmImplementation()
{
	if (m_model)
	{
		svm_free_model_content(m_model);
		//delete m_model;
		m_model = NULL;

		if (m_problem.y != nullptr )
		{
			delete m_problem.y;
			for (auto i = 0; i < m_problem.l; ++i)
			{
				delete m_problem.x[i];
			}
			delete m_problem.x;
		}
	}
}


// libSvmImplementation::libSvmImplementation(std::shared_ptr<svmComponents::IGroupPropagation> strategy)
// 	: libSvmImplementation()
// {
// 	m_groupStrategy = strategy;
// 	m_groupStrategyName = svmComponents::GroupsStrategyFactory::convert(strategy->getType());
// }

libSvmImplementation::libSvmImplementation()
	: m_model(nullptr)
{
	srand(0);
	m_param.svm_type = C_SVC;
	m_param.kernel_type = RBF;
	m_param.degree = 3;
	m_param.gamma = 0.01; // 1/num_features
	m_param.coef0 = 0;
	m_param.cache_size = 200;
	m_param.eps = 0.001; 
	m_param.C = 1;
	m_param.nr_weight = 0;
	m_param.weight_label = NULL;
	m_param.weight = NULL;
	m_param.nu = 0.5;
	m_param.p = 0.1;
	m_param.shrinking = 0;
	m_param.probability = 0;
	m_param.gammas = nullptr;
	m_param.gammas_after_training = nullptr;
	m_param.features = nullptr;
	m_param.reached_max_iter = false;
	m_param.trainAlpha = true;
	m_param.m_optimalThresholdSet = false;

	m_param.m_optimalProbabilityThreshold = 0;
	
	m_isMulticlass = false;
	m_param.m_certaintyPositive = -1111;
	m_param.m_certaintyNegative = -1111;
	m_param.m_certaintyNegativeNormalized = -1111;
	m_param.m_certaintyPositiveNormalized = -1111;


	m_param.m_certaintyPositiveClassOnly = -1111;
	m_param.m_certaintyNegativeClassOnly = -1111;
	m_param.m_certaintyNegativeNormalizedClassOnly = -1111;
	m_param.m_certaintyPositiveNormalizedClassOnly = -1111;
}

libSvmImplementation::libSvmImplementation(const std::filesystem::path& filepath)
{
	srand(0);
	
	m_model = svm_load_model(filepath.string().c_str());

	m_groupStrategyName = std::string(*m_model->groupStrategyName);
	delete m_model->groupStrategyName;
	m_model->groupStrategyName = &m_groupStrategyName;
	//m_groupStrategy = svmComponents::GroupsStrategyFactory::create(svmComponents::GroupsStrategyFactory::convert(m_groupStrategyName));

	if(m_model->param.features)
	{
		m_featureSet = *m_model->param.features;
	}

	if(m_model->param.kernel_type != RBF_CUSTOM
		&& m_model->param.kernel_type != RBF_SUM
		&& m_model->param.kernel_type != RBF_SUM_DIV2
		&& m_model->param.kernel_type != RBF_DIV
		&& m_model->param.kernel_type != RBF_MAX
		&& m_model->param.kernel_type != RBF_MIN
		&& m_model->param.kernel_type != RBF_SUM_2_KERNELS
		&& m_model->param.kernel_type != RBF_LINEAR
		&& m_model->param.kernel_type != RBF_LINEAR_MAX
		&& m_model->param.kernel_type != RBF_LINEAR_MIN
		&& m_model->param.kernel_type != RBF_LINEAR_SUM_2_KERNELS
		&& m_model->param.kernel_type != RBF_LINEAR_SINGLE)
	{
		m_model->param.gammas = nullptr;
		m_model->param.gammas_after_training = nullptr;
		m_model->param.reached_max_iter = false;
	}

	m_param.kernel_type = (m_model->param.kernel_type);
	
	if (m_model->param.kernel_type == RBF_CUSTOM || m_model->param.kernel_type == RBF_SUM
		|| m_model->param.kernel_type == RBF_SUM_DIV2
		|| m_model->param.kernel_type == RBF_DIV
		|| m_model->param.kernel_type == RBF_MAX
		|| m_model->param.kernel_type == RBF_MIN
		|| m_model->param.kernel_type == RBF_SUM_2_KERNELS
		|| m_model->param.kernel_type == RBF_LINEAR
		|| m_model->param.kernel_type == RBF_LINEAR_MAX
		|| m_model->param.kernel_type == RBF_LINEAR_MIN
		|| m_model->param.kernel_type == RBF_LINEAR_SUM_2_KERNELS
		|| m_model->param.kernel_type == RBF_LINEAR_SINGLE)
	{
		m_param.kernel_type = (m_model->param.kernel_type);
		std::set<double> gammas_values;
		for(auto v : *m_model->param.gammas_after_training)
		{
			gammas_values.emplace(v);
		}
		std::copy(gammas_values.begin(), gammas_values.end(), std::back_inserter(m_gammas));
	}
	
	m_problem.y = nullptr;
	m_param.m_optimalThresholdSet = true;
}

//libSvmImplementation::libSvmImplementation(const std::string& model_text)
//{
//	srand(0);
//
//	m_model = svm_load_model_from_string(model_text);
//
//	if (m_model->param.kernel_type != RBF_CUSTOM
//		&& m_model->param.kernel_type != RBF_SUM
//		&& m_model->param.kernel_type != RBF_SUM_DIV2
//		&& m_model->param.kernel_type != RBF_DIV
//		&& m_model->param.kernel_type != RBF_MAX
//		&& m_model->param.kernel_type != RBF_MIN
//		&& m_model->param.kernel_type != RBF_SUM_2_KERNELS
//		&& m_model->param.kernel_type != RBF_LINEAR
//		&& m_model->param.kernel_type != RBF_LINEAR_MAX
//		&& m_model->param.kernel_type != RBF_LINEAR_MIN
//		&& m_model->param.kernel_type != RBF_LINEAR_SUM_2_KERNELS
//		&& m_model->param.kernel_type != RBF_LINEAR_SINGLE)
//	{
//		m_model->param.gammas = nullptr;
//		m_model->param.gammas_after_training = nullptr;
//		m_model->param.reached_max_iter = false;
//	}
//
//	m_param.kernel_type = (m_model->param.kernel_type);
//
//	if (m_model->param.kernel_type == RBF_CUSTOM || m_model->param.kernel_type == RBF_SUM
//		|| m_model->param.kernel_type == RBF_SUM_DIV2
//		|| m_model->param.kernel_type == RBF_DIV
//		|| m_model->param.kernel_type == RBF_MAX
//		|| m_model->param.kernel_type == RBF_MIN
//		|| m_model->param.kernel_type == RBF_SUM_2_KERNELS
//		|| m_model->param.kernel_type == RBF_LINEAR
//		|| m_model->param.kernel_type == RBF_LINEAR_MAX
//		|| m_model->param.kernel_type == RBF_LINEAR_MIN
//		|| m_model->param.kernel_type == RBF_LINEAR_SUM_2_KERNELS
//		|| m_model->param.kernel_type == RBF_LINEAR_SINGLE)
//	{
//		m_param.kernel_type = (m_model->param.kernel_type);
//		std::set<double> gammas_values;
//		for (auto v : *m_model->param.gammas_after_training)
//		{
//			gammas_values.emplace(v);
//		}
//		std::copy(gammas_values.begin(), gammas_values.end(), std::back_inserter(m_gammas));
//	}
//
//	m_problem.y = nullptr;
//	m_optimalThresholdSet = true;
//}

std::string libSvmImplementation::saveToString()
{
	// if (m_groupStrategy)
	// {
	// 	m_groupStrategyName = svmComponents::GroupsStrategyFactory::convert(m_groupStrategy->getType());
	// 	m_model->groupStrategyName = &m_groupStrategyName;
	// }
	m_model->param.features = &m_featureSet;
	return svm_save_model_to_string(m_model);
}

void libSvmImplementation::loadFromString(const std::string& model_text)
{
		srand(0);

	m_model = svm_load_model_from_string(model_text);

	m_groupStrategyName = std::string(*m_model->groupStrategyName);
	delete m_model->groupStrategyName;
	m_model->groupStrategyName = &m_groupStrategyName;
	//m_groupStrategy = svmComponents::GroupsStrategyFactory::create(svmComponents::GroupsStrategyFactory::convert(m_groupStrategyName));

	if(m_model->param.features)
	{
		std::vector<svmComponents::Feature> features;
		for (auto v : *m_model->param.features)
		{
			features.emplace_back(v);
		}
		std::copy(features.begin(), features.end(), std::back_inserter(m_featureSet));
	}
	
	if (m_model->param.kernel_type != RBF_CUSTOM
		&& m_model->param.kernel_type != RBF_SUM
		&& m_model->param.kernel_type != RBF_SUM_DIV2
		&& m_model->param.kernel_type != RBF_DIV
		&& m_model->param.kernel_type != RBF_MAX
		&& m_model->param.kernel_type != RBF_MIN
		&& m_model->param.kernel_type != RBF_SUM_2_KERNELS
		&& m_model->param.kernel_type != RBF_LINEAR
		&& m_model->param.kernel_type != RBF_LINEAR_MAX
		&& m_model->param.kernel_type != RBF_LINEAR_MIN
		&& m_model->param.kernel_type != RBF_LINEAR_SUM_2_KERNELS
		&& m_model->param.kernel_type != RBF_LINEAR_SINGLE)
	{
		m_model->param.gammas = nullptr;
		m_model->param.gammas_after_training = nullptr;
		m_model->param.reached_max_iter = false;
	}

	m_param.kernel_type = (m_model->param.kernel_type);

	if (m_model->param.kernel_type == RBF_CUSTOM || m_model->param.kernel_type == RBF_SUM
		|| m_model->param.kernel_type == RBF_SUM_DIV2
		|| m_model->param.kernel_type == RBF_DIV
		|| m_model->param.kernel_type == RBF_MAX
		|| m_model->param.kernel_type == RBF_MIN
		|| m_model->param.kernel_type == RBF_SUM_2_KERNELS
		|| m_model->param.kernel_type == RBF_LINEAR
		|| m_model->param.kernel_type == RBF_LINEAR_MAX
		|| m_model->param.kernel_type == RBF_LINEAR_MIN
		|| m_model->param.kernel_type == RBF_LINEAR_SUM_2_KERNELS
		|| m_model->param.kernel_type == RBF_LINEAR_SINGLE)
	{
		m_param.kernel_type = (m_model->param.kernel_type);
		std::set<double> gammas_values;
		for (auto v : *m_model->param.gammas_after_training)
		{
			gammas_values.emplace(v);
		}
		std::copy(gammas_values.begin(), gammas_values.end(), std::back_inserter(m_gammas));
	}

	m_problem.y = nullptr;
}

libSvmImplementation::libSvmImplementation(const std::filesystem::path& filepath, const dataset::Dataset<std::vector<float>, float>& trainingSet)
{
	srand(0);

	m_model = svm_load_model(filepath.string().c_str());

	m_groupStrategyName = std::string(*m_model->groupStrategyName);
	delete m_model->groupStrategyName;
	m_model->groupStrategyName = &m_groupStrategyName;
	//m_groupStrategy = svmComponents::GroupsStrategyFactory::create(svmComponents::GroupsStrategyFactory::convert(m_groupStrategyName));

	if (m_model->param.features)
	{
		std::vector<svmComponents::Feature> features;
		for (auto v : *m_model->param.features)
		{
			features.emplace_back(v);
		}
		std::copy(features.begin(), features.end(), std::back_inserter(m_featureSet));
	}
	
	if (m_model->param.kernel_type != RBF_CUSTOM
		&& m_model->param.kernel_type != RBF_SUM
		&& m_model->param.kernel_type != RBF_SUM_DIV2
		&& m_model->param.kernel_type != RBF_DIV
		&& m_model->param.kernel_type != RBF_MAX
		&& m_model->param.kernel_type != RBF_MIN
		&& m_model->param.kernel_type != RBF_SUM_2_KERNELS
		&& m_model->param.kernel_type != RBF_LINEAR
		&& m_model->param.kernel_type != RBF_LINEAR_MAX
		&& m_model->param.kernel_type != RBF_LINEAR_MIN
		&& m_model->param.kernel_type != RBF_LINEAR_SUM_2_KERNELS
		&& m_model->param.kernel_type != RBF_LINEAR_SINGLE)
	{
		m_model->param.gammas = nullptr;
		m_model->param.gammas_after_training = nullptr;
		m_model->param.reached_max_iter = false;
	}

	m_param.kernel_type = (m_model->param.kernel_type);

	if (m_model->param.kernel_type == RBF_CUSTOM || m_model->param.kernel_type == RBF_SUM
		|| m_model->param.kernel_type == RBF_SUM_DIV2
		|| m_model->param.kernel_type == RBF_DIV
		|| m_model->param.kernel_type == RBF_MAX
		|| m_model->param.kernel_type == RBF_MIN
		|| m_model->param.kernel_type == RBF_SUM_2_KERNELS
		|| m_model->param.kernel_type == RBF_LINEAR
		|| m_model->param.kernel_type == RBF_LINEAR_MAX
		|| m_model->param.kernel_type == RBF_LINEAR_MIN
		|| m_model->param.kernel_type == RBF_LINEAR_SUM_2_KERNELS
		|| m_model->param.kernel_type == RBF_LINEAR_SINGLE)
	{
		m_param.kernel_type = (m_model->param.kernel_type);
		std::set<double> gammas_values;
		for (auto v : *m_model->param.gammas_after_training)
		{
			gammas_values.emplace(v);
		}
		std::copy(gammas_values.begin(), gammas_values.end(), std::back_inserter(m_gammas));
	}

	setupSupportVectorLoadedFromFile(trainingSet);

	m_problem.y = nullptr;
}

bool libSvmImplementation::isMaxIterReached() const
{
	return m_param.reached_max_iter;
}


std::vector<int> libSvmImplementation::getSvLables() const
{
	std::vector<int> labels;

	for(auto i = 0; i < m_model->l; ++i)
	{
		//std::cout << m_model->sv_coef[0][i] << ", ";
		if (m_model->sv_coef[0][i] > 0)
		{
			labels.emplace_back(1);
		}
		else
		{
			labels.emplace_back(0);
		}
	}
	//std::cout << "\n";

	return labels;
}

std::vector<double> libSvmImplementation::getGammas() const
{
	return *m_model->param.gammas_after_training;
}

KernelTypes libSvmImplementation::getKernelType() const
{
	switch (m_param.kernel_type)
	{
	case RBF:
		return KernelTypes::Rbf;
	case LINEAR:
		return KernelTypes::Linear;
	case RBF_CUSTOM:
		return KernelTypes::Rbf_custom;
	case RBF_SUM:
		return KernelTypes::RBF_SUM;
	case RBF_SUM_DIV2:
		return KernelTypes::RBF_SUM_DIV2;
	case RBF_DIV:
		return KernelTypes::RBF_DIV;
	case RBF_MAX:
		return KernelTypes::RBF_MAX;
	case RBF_MIN:
		return  KernelTypes::RBF_MIN;
	case RBF_SUM_2_KERNELS:
		return KernelTypes::RBF_SUM_2_KERNELS;
	case RBF_LINEAR:
		return KernelTypes::RBF_LINEAR;
	case RBF_LINEAR_MIN:
		return KernelTypes::RBF_LINEAR_MIN;
	case RBF_LINEAR_SINGLE:
		return KernelTypes::RBF_LINEAR_SINGLE;
	case RBF_LINEAR_MAX:
		return KernelTypes::RBF_LINEAR_MAX;
	case RBF_LINEAR_SUM_2_KERNELS:
		return KernelTypes::RBF_LINEAR_SUM_2_KERNELS;
	case RBF_POLY_GLOBAL:
		return KernelTypes::RBF_POLY_GLOBAL;
	default:
		return KernelTypes::Custom;
	}
}

void libSvmImplementation::setKernel(KernelTypes kernelType)
{
	switch (kernelType)
	{
	case KernelTypes::Custom: m_param.kernel_type = PRECOMPUTED;
		break;
	case KernelTypes::Linear: m_param.kernel_type = LINEAR;
		break;
	case KernelTypes::Poly: m_param.kernel_type = POLY;
		break;
	case KernelTypes::Rbf: m_param.kernel_type = RBF;
		break;
	case KernelTypes::Sigmoid: m_param.kernel_type = SIGMOID;
		break;
	case KernelTypes::Rbf_custom: m_param.kernel_type = RBF_CUSTOM;
		break;
	case KernelTypes::RBF_SUM:
		m_param.kernel_type = RBF_SUM;
		break;
	case KernelTypes::RBF_SUM_DIV2:
		m_param.kernel_type = RBF_SUM_DIV2;
		break;
	case KernelTypes::RBF_DIV:
		m_param.kernel_type = RBF_DIV;
		break;
	case KernelTypes::RBF_MAX:
		m_param.kernel_type = RBF_MAX;
		break;
	case KernelTypes::RBF_MIN:
		m_param.kernel_type = RBF_MIN;
		break;
	case KernelTypes::RBF_SUM_2_KERNELS:
		m_param.kernel_type = RBF_SUM_2_KERNELS;
		break;
	case KernelTypes::RBF_LINEAR:
		m_param.kernel_type = RBF_LINEAR;
		break;
	case KernelTypes::RBF_LINEAR_SINGLE:
		m_param.kernel_type = RBF_LINEAR_SINGLE;
		break;
	case KernelTypes::RBF_LINEAR_MIN:
		m_param.kernel_type = RBF_LINEAR_MIN;
		break;
	case KernelTypes::RBF_LINEAR_MAX:
		m_param.kernel_type =  RBF_LINEAR_MAX;
		break;
	case KernelTypes::RBF_LINEAR_SUM_2_KERNELS:
		m_param.kernel_type =  RBF_LINEAR_SUM_2_KERNELS;
		break;
	case KernelTypes::RBF_POLY_GLOBAL:
		m_param.kernel_type = RBF_POLY_GLOBAL;
		break;
		/*  case KernelTypes::Chi2: m_param.kernel_type =  break;
		  case KernelTypes::Inter: m_param.kernel_type =  break;*/
	default: m_param.kernel_type = 1000;
	}
}

SvmTypes libSvmImplementation::getType() const
{
	switch (m_param.svm_type)
	{
	case C_SVC:
		return SvmTypes::CSvc;
	case EPSILON_SVR:
		return SvmTypes::EpsSvr;
	case NU_SVC:
		return SvmTypes::NuSvc;
	default:
		return SvmTypes::Unknown;
	}
}

void libSvmImplementation::setType(SvmTypes svmType)
{
	switch (svmType)
	{
	case SvmTypes::Unknown: break;
	case SvmTypes::CSvc: m_param.svm_type = C_SVC;
		break;
	case SvmTypes::NuSvc: m_param.svm_type = NU_SVC;
		break;
	case SvmTypes::OneClass: m_param.svm_type = ONE_CLASS;
		break;
	case SvmTypes::EpsSvr: m_param.svm_type = EPSILON_SVR;
		break;
	case SvmTypes::NuSvr: m_param.svm_type = NU_SVR;
		break;
	default: ;
	}
}

void libSvmImplementation::save(const std::filesystem::path& filepath)
{
	// if(m_groupStrategy)
	// {
	// 	m_groupStrategyName = svmComponents::GroupsStrategyFactory::convert(m_groupStrategy->getType());
	// 	m_model->groupStrategyName = &m_groupStrategyName;
	// }
	if(!m_featureSet.empty())
	{
		m_model->param.features = &m_featureSet;
	}
	svm_save_model(filepath.string().c_str(), m_model);
}

//TODO delete this function as it is not used
uint32_t libSvmImplementation::getNumberOfKernelParameters(KernelTypes kernelType) const
{
	switch (kernelType)
	{
	case KernelTypes::Rbf:
		return 2;
	default:
		throw UnsupportedKernelTypeException(kernelType);
	}
}

uint32_t libSvmImplementation::getNumberOfSupportVectors() const
{
	return svm_get_nr_sv(m_model);
}

// cv::Mat libSvmImplementation::getSupportVectors() const
// {
// 	return m_sv;
// }

std::vector<std::vector<float>> libSvmImplementation::getSupportVectors() const
{
	return m_sv;
}

void libSvmImplementation::setupSupportVector(const dataset::Dataset<std::vector<float>, float>& trainingSet)
{
	auto featureNumber = trainingSet.getSamples()[0].size();
	auto samples = trainingSet.getSamples();

	//cv::Mat m = cv::Mat::zeros(0, static_cast<int>(featureNumber), CV_32F);
	std::vector<std::vector<float>> m;


	auto nr_sv = svm_get_nr_sv(m_model);
	std::vector<int> sv_indices;
	sv_indices.resize(nr_sv);
	svm_get_sv_indices(m_model, sv_indices.data());
	for (int i = 0; i < nr_sv; i++)
	{
		std::vector<float> retirved_sv;
		retirved_sv.resize(featureNumber);
		auto sv = m_model->SV[i];
		while (sv->index != -1)
		{
			retirved_sv[sv->index] = (static_cast<float>(sv->value));
			sv++;
		}
		//printf("instance %d is a support vector\n", sv_indices[i]);
		// cv::Mat row = cv::Mat(1, static_cast<int>(featureNumber), CV_32F, const_cast<float*>(samples[sv_indices[i] - 1].data()));
		// m.push_back(row);
		m.emplace_back(retirved_sv);
	}

	m_sv = m;
}

double libSvmImplementation::getT() const
{
	return m_param.t;
}

void libSvmImplementation::setT(double value)
{
	m_param.t = value;
}

double libSvmImplementation::getMinSvDistance()
{
	//cv::Mat vectors = getSupportVectors();

	std::vector<double> distances; 
	
	auto nr_sv = svm_get_nr_sv(m_model);
	for (int i = 0; i < nr_sv; i++)
	{
		for (int j = i + 1; j < nr_sv; j++)
		{
			auto dist = vectorDistance(m_model->SV[i], m_model->SV[j]);
			distances.emplace_back(dist);
		}
	}

	return *std::min_element(distances.begin(), distances.end());
}

std::vector<uint64_t> libSvmImplementation::getCertaintyRegion(const dataset::Dataset<std::vector<float>, float>& dataset)
{
	LOG_F(WARNING, "wrong method for certainty region called");
	
	auto& svmModel = *this;
	auto targets = dataset.getLabels();
	auto samples = dataset.getSamples();

	std::vector<uint64_t> samplesIds;

	//hyperplane distance, classify result, true value
	std::vector<std::tuple<double, int, int, unsigned int>> results;
	results.reserve(targets.size());

	for (auto i = 0u; i < targets.size(); i++)
	{
		auto temp = std::make_tuple(static_cast<double>(svmModel.classifyHyperplaneDistance(samples[i])),
			static_cast<int>(svmModel.classify(samples[i])),
			static_cast<int>(targets[i]),
			static_cast<unsigned int>(i));

		results.emplace_back(temp);
	}

	/* std::vector<std::tuple<double, int, int>> results_copy;
	 std::copy(results.begin(), results.end(), std::back_inserter(results_copy));*/

	std::sort(results.begin(), results.end(),
		[&](const std::tuple<double, int, int, unsigned int>& a, const std::tuple<double, int, int, unsigned int>& b)
		{
			return std::get<0>(a) < std::get<0>(b);
		});


	double max_distance = 0.0;
	double min_distance = 1000000.0;
	for (auto i = 0u; i < results.size(); ++i)
	{
		if (std::get<1>(results[i]) != std::get<2>(results[i])) //if classify == target
		{
			break;
		}
		min_distance = std::get<0>(results[i]) / std::get<0>(results[0]);
		samplesIds.emplace_back(std::get<3>(results[i]));
	}

	for (auto i = results.size() - 1; i > 0; --i)
	{
		if (std::get<1>(results[i]) != std::get<2>(results[i])) //if classify == target
		{
			break;
		}
		max_distance = std::get<0>(results[i]) / std::get<0>(results[results.size() - 1]);
		samplesIds.emplace_back(std::get<3>(results[i]));
	}

	return samplesIds;
}

std::vector<uint64_t> libSvmImplementation::getUncertaintyRegion(const dataset::Dataset<std::vector<float>, float>& dataset)
{
	LOG_F(WARNING, "wrong method for uncertainty region called");
	
	auto samplesIds = getCertaintyRegion(dataset);

	std::sort(samplesIds.begin(), samplesIds.end());
	
	std::vector<uint64_t> result; 
	std::vector<uint64_t> allIds(dataset.getSamples().size());
	std::iota(std::begin(allIds), std::end(allIds), 0); // Fill with 0, 1, 2.....
	std::set_difference(allIds.begin(), allIds.end(), samplesIds.begin(), samplesIds.end(), std::back_inserter(result));
		
	return result;
}

void libSvmImplementation::setupSupportVectorLoadedFromFile(const dataset::Dataset<std::vector<float>, float>& trainingSet)
{
	auto featureNumber = trainingSet.getSamples()[0].size();
	auto samples = trainingSet.getSamples();

	//cv::Mat m = cv::Mat::zeros(0, static_cast<int>(featureNumber), CV_32F);

	std::vector<std::vector<float>> m;

	auto nr_sv = svm_get_nr_sv(m_model);
	std::vector<int> sv_indices;
	sv_indices.resize(nr_sv);
	svm_get_sv_indices(m_model, sv_indices.data());
	for (int i = 0; i < nr_sv; i++)
	{
		std::vector<float> retirved_sv;
		retirved_sv.resize(featureNumber);
		auto sv = m_model->SV[i];
		while (sv->index != -1)
		{
			retirved_sv[sv->index] = (static_cast<float>(sv->value));
			sv++;
		}

		//cv::Mat row = cv::Mat(1, static_cast<int>(featureNumber), CV_32F, const_cast<float*>(retirved_sv.data()));
		//m.push_back(row);

		m.emplace_back(retirved_sv);
	}

	m_sv = m;
}

float libSvmImplementation::classify(const gsl::span<const float> sample) const
{
	auto node = convertSample(sample);

	return static_cast<float>(svm_predict(m_model, &node[0]));
}

void libSvmImplementation::train(const dataset::Dataset<std::vector<float>, float>& trainingSet, bool probabilityNeeded)
{
	if (probabilityNeeded)
	{
		m_param.probability = 1;
	}

	m_problem = createDatasetForTraining(trainingSet);

	m_model = svm_train(&m_problem, &m_param);
	m_model->param.m_optimalProbabilityThreshold = 0; 

	setupSupportVector(trainingSet);
}

//Better not to use platt scalling and classification probability
double libSvmImplementation::classificationProbability(const gsl::span<const float> sample) const
{
	assert(false && "DO NOT USE THIS FUNCTION");
	
	auto node = convertSample(sample);
	
	std::vector<double> probs(m_model->nr_class);

	svm_predict_probability(m_model, &node[0], probs.data());

	return probs[1]; //TODO check this probs[m_model->label[1]];
}

// void libSvmImplementation::setTerminationCriteria(const cv::TermCriteria& value)
// {
// 	m_param.eps = value.epsilon;
// }

// cv::TermCriteria libSvmImplementation::getTerminationCriteria() const
// {
// 	return cv::TermCriteria(cv::TermCriteria::EPS, 100, m_param.eps);
// }

bool libSvmImplementation::isTrained() const
{
	return m_model != nullptr;
}

bool libSvmImplementation::canGiveProbabilityOutput() const
{
	return svm_check_probability_model(m_model) == 1;
}

double libSvmImplementation::classifyWithOptimalThreshold(const gsl::span<const float> sample) const
{
	/*if(m_isMulticlass)
	{
		return classify(sample);
	}*/
	
	if(m_model->param.m_optimalThresholdSet)
	{
		return classifyHyperplaneDistance(sample) > m_model->param.m_optimalProbabilityThreshold ? 1.0 : 0.0;
	}
	else
	{
		return classify(sample);
		//throw std::exception("Optimal threshold not set");
	}
}

void libSvmImplementation::setOptimalProbabilityThreshold(double optimalThreshold)
{
	m_model->param.m_optimalThresholdSet = true;
	m_model->param.m_optimalProbabilityThreshold = optimalThreshold;
	
	m_param.m_optimalProbabilityThreshold = optimalThreshold;
	m_param.m_optimalThresholdSet = true;
}

bool libSvmImplementation::canClassifyWithOptimalThreshold() const
{
	return m_model->param.m_optimalThresholdSet;
}

void libSvmImplementation::setFeatureSet(const std::vector<svmComponents::Feature>& features, int numberOfFeatures)
{
	m_featureSet = features;
	m_numberOfFeatures = numberOfFeatures;
	//m_model->param.features = &m_featureSet;
	m_param.features = &m_featureSet;
}

const std::vector<svmComponents::Feature>& libSvmImplementation::getFeatureSet()
{
	return m_featureSet;
}

float libSvmImplementation::classifyDistanceToClosestSV(const gsl::span<const float> sample) const
{
	auto node = convertSample(sample);

	const auto nr_class = m_model->nr_class;
	double* decValues = new double[nr_class * (nr_class - 1) / 2];
	
	auto value = svm_predict_values_with_closest_distance(m_model, &node[0], decValues);
	
	const auto temp = decValues[0]; //TODO make it work for mutliple class
	delete[] decValues;

	//it happen that raw decision value have different sign than label, 
	//it caused problem with AUC calculation(value < 0.5) and was changing the visualizations
	/*if ((value.first == 0 && temp > 0) || (value.second == 1 && temp < 0))
	{
		return static_cast<float>(-temp);
	}
	return static_cast<float>(temp);*/
	return static_cast<float>(value.second);
}

float libSvmImplementation::classifyHyperplaneDistance(const gsl::span<const float> sample) const
{
	auto node = convertSample(sample);

	//Will not work for more than 100 classes
	//std::array<double, 100> decValues2;
	//auto value = svm_predict_values(m_model, &node[0], decValues2.data());
	//const auto temp = decValues2[0]; //TODO make it work for mutliple class
	
	const auto nr_class = m_model->nr_class;
	double* decValues;
	if (m_model->param.svm_type == ONE_CLASS ||
		m_model->param.svm_type == EPSILON_SVR ||
		m_model->param.svm_type == NU_SVR)
		decValues = new double[1];
	else
		decValues = new double[nr_class * (nr_class - 1) / 2];
	auto value = svm_predict_values(m_model, &node[0], decValues);
	//svm_predict_values(m_model, &node[0], decValues);
	
	const auto temp = decValues[0]; //TODO make it work for mutliple class
	delete[] decValues;

	//it happen that raw decision value have different sign than label, 
	//it caused problem with AUC calculation(value < 0.5) and was changing the visualizations
	if ((value == 0 && temp > 0) || (value == 1 && temp < 0))
	{
		return static_cast<float>(-temp);
	}

	//note that in libSVM the result of DecisionValue should be divided by |w| to obtain distance
	//but in our case for classification purpose dividing by constant would not change the results (sign of function)
	//This will also not affect the AUC score calculated 
	//for calculating |w| see https://www.csie.ntu.edu.tw/~cjlin/libsvm/faq.html#f4151
	return static_cast<float>(temp);
}

std::tuple<double,double,int> libSvmImplementation::classifyPositiveNegative(const gsl::span<const float> sample) const
{
	auto node = convertSample(sample);
	auto [pos, neg, label] =  svm_predict_values_pos_neg(m_model, &node[0]);

	//auto result = classifyHyperplaneDistance(sample);
	

	return std::make_tuple(pos, neg, label);
}


float libSvmImplementation::classifyWithCertainty(const gsl::span<const float> sample) const
{
	if(m_model->param.m_certaintyPositive == -1111 && m_model->param.m_certaintyNegative == -1111)
	{
		//LOG_F(INFO, "no certainty using optimal threshold");
		return static_cast<float>(classifyWithOptimalThreshold(sample));
	}
	//used in case when class boundaries are taken into account 
	if(m_model->param.m_certaintyNegativeClassOnly != -1111 && m_model->param.m_certaintyPositiveClassOnly != -1111)
	{
		auto [pos, neg, label] = classifyPositiveNegative(sample);
		auto sum = pos + neg - m_model->rho[0];
		
		if (sum > m_model->param.m_certaintyPositive && pos > m_model->param.m_certaintyPositiveClassOnly)
		{
			return 1; //positive
		}
		else if (sum < m_model->param.m_certaintyNegative && neg < m_model->param.m_certaintyNegativeClassOnly)
		{
			return 0; //unknown
		}
		else
		{
			return -100; //negative
		}
	}

	auto result = classifyHyperplaneDistance(sample);
	
	if( result > m_model->param.m_certaintyPositive )
	{
		return 1; //positive
	}
	else if( result < m_model->param.m_certaintyNegative)
	{
		return 0; //unknown
	}
	else
	{
		return -100; //negative
	}
}

double vectorDistance(svm_node* first, svm_node* second) {
	double ret = 0.0;
	

	while (first->index != -1 && second->index != -1)
	{
		if (first->index == second->index)
		{
			ret += std::pow(first->value - second->value, 2);
			++first;
			++second;
		}
		else
		{
			//cases where one of index is empty (means 0 value so distance is calcuated appropriately)
			if (first->index > second->index)
			{
				ret += std::pow(first->value, 2);
				++second;
			}
			else
			{
				ret += std::pow(second->value, 2);
				++first;
			}
		}
	}

	return ret > 0.0 ? sqrt(ret) : 0.0;
}

std::tuple<std::multimap<int, int>, std::vector<double>> libSvmImplementation::check_sv(const dataset::Dataset<std::vector<float>, float>& validation)
{
	auto samples = validation.getSamples();
	auto labels = validation.getLabels();

	std::multimap<int, std::pair<double, int>> sv_index_to_value;
	std::vector<double> maxDistances;
	maxDistances.resize(validation.size(), double(0));


	for (auto s = 0; s < m_model->l; ++s)
	{
		for (auto i = 0; i < samples.size(); ++i)
		{
			auto result = convertSample(samples[i]);
			auto distance = Kernel::k_function(&result[0], m_model->SV[s], m_model->param, s)  * fabs(m_model->sv_coef[0][s]);
			sv_index_to_value.insert({ s, {distance, i} }); //[s] = distance;
			if (distance > maxDistances[i])
			{
				maxDistances[i] = distance;
			}
		}
	}

	std::multimap<int, int> sv_index_to_validation_id_prunned;
	for (auto s = 0; s < m_model->l; ++s)
	{
		auto range = sv_index_to_value.equal_range(s);
		int i = 0;
		for (auto it = range.first; it != range.second; ++it)
		{
			if(/*it->second.first == maxDistances[i] ||*/ it->second.first > std::fabs(*m_model->rho))   //it->second.first > std::fabs(*m_model->rho)
			{
				sv_index_to_validation_id_prunned.insert({ s, it->second.second });
			}
			++i;
		}
	}

	std::vector<int> predictions;
	for(auto i = 0; i < samples.size(); ++i)
	{
		 if(classify(samples[i]) == labels[i])
		 {
			 predictions.emplace_back(static_cast<int>(labels[i]));
		 }
		 else
		 {
			 predictions.emplace_back(-1);
		 }
	}

	//scores for sv
	std::vector<double> scoresSV;
	std::vector<double> numberOfExamplesPerSV;
	scoresSV.resize(m_model->l);
	numberOfExamplesPerSV.resize(m_model->l);
	for(auto value : sv_index_to_validation_id_prunned)
	{
		if(predictions[value.second] == labels[value.second])
		{
			scoresSV[value.first]++;
		}
		numberOfExamplesPerSV[value.first]++;
	}

	for(auto i = 0u; i < scoresSV.size(); ++i)
	{
		if(numberOfExamplesPerSV[i] == 0)
		{
			scoresSV[i] = 0;
		}
		else
		{
			scoresSV[i] /= numberOfExamplesPerSV[i];
		}
		
	}

	std::vector<double> minSV_Distances;
	for (auto i = 0; i < m_model->l; ++i)
	{
		auto minDistance = 9999999999999.0;
		for (auto j = 0; j < m_model->l; ++j)
		{
			if (i == j)
				continue;
			auto distance = vectorDistance(m_model->SV[i], m_model->SV[j]);
			if(distance < minDistance)
			{
				minDistance = distance;
			}
		}
		minSV_Distances.push_back(minDistance);
	}

	/*int i = 0;
	for(auto& s : scoresSV)
	{
		if (s >= 0.9)
		{
			s += minSV_Distances[i];
		}
		++i;
	}*/


	/*for(auto values : sv_index_to_value_prunned)
	{
		std::cout << "id: " << values.first << "   value: " << values.second << "\n";
	}*/

	return std::tuple<std::multimap<int,int>, std::vector<double>>{ sv_index_to_validation_id_prunned, scoresSV };
}

void libSvmImplementation::setAlphaTraining(bool value)
{
	m_param.trainAlpha = value;
}

std::vector<svm_node> libSvmImplementation::convertSample(gsl::span<const float> sample) const
{
	/*auto max_id = *std::max_element(m_featureSet.begin(), m_featureSet.end());
	
	if(static_cast<int>(sample.size()) != static_cast<int>(m_featureSet.size()) 
		|| max_id.id  <= static_cast<std::uint64_t>(sample.size()))*/
		//if(static_cast<int>(sample.size()) != static_cast<int>(m_featureSet.size()))
	if(!m_featureSet.empty())
	{
		auto f_copy = m_featureSet;
		std::sort(f_copy.begin(), f_copy.end());

		std::vector<svm_node> node(m_featureSet.size() + 1);
		for (auto i = 0u; i < m_featureSet.size(); i++)
		{
			node[i].index = i;
			node[i].value = sample[m_featureSet[i].id]; //features are in the same order as in chromosome (not ordered)
			//node[i].value = sample[f_copy[i].id];
		}
		node[m_featureSet.size()].index = -1;
		node[m_featureSet.size()].value = 0;

		return node;	
	}
	
	std::vector<svm_node> node(sample.size() + 1);
	for (auto i = 0u; i < sample.size(); i++)
	{
		node[i].index = i;
		node[i].value = sample[i];
	}
	node[sample.size()].index = -1;
	node[sample.size()].value = 0;

	return node;
}

svm_problem libSvmImplementation::createDatasetForTraining(const dataset::Dataset<std::vector<float>, float>& trainingSet)
{
	svm_problem problem;

	// Set size of train set.
	problem.l = static_cast<int>(trainingSet.size());

	auto mY = trainingSet.getLabels();
	auto mX = trainingSet.getSamples();

	assert(mX.size() == mY.size());

	// Set labels of data
	problem.y = new double[problem.l];
	for (unsigned i = 0; i < mY.size(); i++)
	{
		problem.y[i] = mY[i];
	}

	// Set data
	unsigned int dims = static_cast<unsigned>(mX[0].size());
	problem.x = new svm_node*[problem.l];
	for (unsigned i = 0; i < mX.size(); i++)
	{
		problem.x[i] = new svm_node[dims + 1];
		int filledIndex = 0;
		for (unsigned j = 0; j < dims; j++)
		{
			if (mX[i][j] != 0)
			{
				problem.x[i][filledIndex].index = j;
				problem.x[i][filledIndex].value = mX[i][j];
				filledIndex++;
			}
		}
		problem.x[i][filledIndex].index = -1;
		problem.x[i][filledIndex].value = 0;
	}

	return problem;
}

void libSvmImplementation::setCertaintyThreshold(double negative, double positive, double normalizedNegative, double normalizedPositive)
{
	if (std::isnan(normalizedNegative))
	{
		normalizedNegative = -1111111111111;
	}
	if (std::isnan(normalizedPositive))
	{
		normalizedPositive = 1111111111111;
	}
	
	
	m_model->param.m_certaintyNegative = negative;
	m_model->param.m_certaintyPositive = positive;
	m_model->param.m_certaintyNegativeNormalized = normalizedNegative;
	m_model->param.m_certaintyPositiveNormalized = normalizedPositive;

	m_param.m_certaintyNegative = negative;
	m_param.m_certaintyPositive = positive;
	m_param.m_certaintyNegativeNormalized = normalizedNegative;
	m_param.m_certaintyPositiveNormalized = normalizedPositive;
}

void libSvmImplementation::setClassCertaintyThreshold(double negative, double positive, double normalizedNegative, double normalizedPositive)
{
	if (std::isnan(normalizedNegative))
	{
		normalizedNegative = -1111111111111;
	}
	if (std::isnan(normalizedPositive))
	{
		normalizedPositive = 1111111111111;
	}


	m_model->param.m_certaintyNegativeClassOnly = negative;
	m_model->param.m_certaintyPositiveClassOnly = positive;
	m_model->param.m_certaintyNegativeNormalizedClassOnly = normalizedNegative;
	m_model->param.m_certaintyPositiveNormalizedClassOnly = normalizedPositive;

	m_param.m_certaintyNegativeClassOnly = negative;
	m_param.m_certaintyPositiveClassOnly = positive;
	m_param.m_certaintyNegativeNormalizedClassOnly = normalizedNegative;
	m_param.m_certaintyPositiveNormalizedClassOnly = normalizedPositive;
}

double libSvmImplementation::getPositiveCertainty() const
{
	return m_model->param.m_certaintyPositive;
}

double libSvmImplementation::getNegativeCertainty() const
{
	return m_model->param.m_certaintyNegative;
}

double libSvmImplementation::getPositiveNormalizedCertainty() const
{
	return m_model->param.m_certaintyPositiveNormalized;
}

double libSvmImplementation::getNegativeNormalizedCertainty() const
{
	return m_model->param.m_certaintyNegativeNormalized;
}
}} // namespace phd::svm
