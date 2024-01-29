#include "libPlatform/EnumStringConversions.h"
#include "SvmLib/EnumsTranslations.h"
#include "SvmComponentsExceptions.h"
#include "SvmUtils.h"
#include "SvmConfigStructures.h"
#include "SvmMetricFactory.h"
#include "RbfKernel.h"
#include "SvmKernelTraining.h"
#include "LinearKernel.h"
#include "PolyKernel.h"
#include "RbfPolyGlobalKernel.h"

namespace svmComponents
{
const std::unordered_map<std::string, svmMetricType> SvmAlgorithmConfiguration::m_translations =
{
	{"Accuracy", svmMetricType::Accuracy}, //ugly hack for quick test
	{"AUC", svmMetricType::Auc},
	{"R2", svmMetricType::R2},
	{"PRAUC", svmMetricType::PrAuc},
	{"BalancedAccuracy",svmMetricType::BalancedAccuracy},
	{"HyperplaneDistance",svmMetricType::HyperplaneDistance},
	{"MCC", svmMetricType::MCC},
	{"CertainAccuracy",svmMetricType::CertainAccuracy},
};

SvmAlgorithmConfiguration::SvmAlgorithmConfiguration(const platform::Subtree& config)
	: m_kernelType([&config]()
	{
		auto kernelName = config.getValue<std::string>("Svm.KernelType");
		auto kernelType = phd::svm::stringToKernelType(kernelName);
		return kernelType;
	}())
	, m_svmEpsilon(config.getValue<double>("Svm.Epsilon"))
	, m_useSvmIteration(config.getValue<bool>("Svm.UseSvmIteration"))
	, m_svmIterationNumber(config.getValue<int>("Svm.SvmIterationNumber"))
	, m_estimationType([&config]()
	{
		auto name = config.getValue<std::string>("Svm.Metric");
		return platform::stringToEnum(name, m_translations);
	}())
	, m_estimationMethod(SvmMetricFactory::create(m_estimationType))
	//, m_groupPropagationMethod(std::make_shared<svmComponents::AverageAnswerPropagation>())
	, m_doVisualization(config.getValue<bool>("Svm.Visualization.Create"))
	, m_height(config.getValue<int>("Svm.Visualization.Height"))
	, m_width(config.getValue<int>("Svm.Visualization.Width"))
	, m_visualizationFormat(imageFormat::png)
	, m_implementationType(phd::svm::SvmFactory::implementationTypeFromString(config.getValue<std::string>("Svm.Type")))

{
}

GridSearchConfiguration::GridSearchConfiguration(const platform::Subtree& config)
	: m_svmConfig(config)
	, m_numberOfFolds(config.getValue<int>("GridSearch.NumberOfFolds"))
	, m_numberOfIterations(config.getValue<int>("GridSearch.NumberOfIteratrions"))
	, m_subsetSize(config.getValue<int>("GridSearch.SubsetSize"))
	, m_subsetIterations(config.getValue<int>("GridSearch.SubsetRepeats"))
	, m_training(std::make_shared<SvmKernelTraining>(m_svmConfig, m_svmConfig.m_estimationType == svmMetricType::Auc))
	, m_kernel(createKernelGrid(m_svmConfig.m_kernelType, config))
{
}

GridSearchConfiguration::GridSearchConfiguration(SvmAlgorithmConfiguration svmConfig,
                                                 unsigned numberOfFolds,
                                                 unsigned numberOfIterations,
                                                 unsigned subsetSize,
                                                 unsigned subsetIterations,
                                                 std::shared_ptr<ISvmTraining<SvmKernelChromosome>> training,
                                                 std::shared_ptr<BaseKernelGridSearch> kernel)
	: m_svmConfig(svmConfig)
	, m_numberOfFolds(numberOfFolds)
	, m_numberOfIterations(numberOfIterations)
	, m_subsetSize(subsetSize)
	, m_subsetIterations(subsetIterations)
	, m_training(std::move(training))
	, m_kernel(std::move(kernel))
{
}

ParamGrid GridSearchConfiguration::parseGridParameters(const std::string& gridName, const platform::Subtree& config)
{
	static const auto emptyGrid = ParamGrid(0, 0, 1);
	validateGridName(gridName);
	if (config.contains("GridSearch." + gridName + ".Min"))
	{
		auto gridMin = config.getValue<double>("GridSearch." + gridName + ".Min");
		auto gridMax = config.getValue<double>("GridSearch." + gridName + ".Max");
		auto gridLogStep = config.getValue<double>("GridSearch." + gridName + ".LogStep");
		return ParamGrid(gridMin, gridMax, gridLogStep);
	}
	return emptyGrid;
}

std::shared_ptr<BaseKernelGridSearch> GridSearchConfiguration::createKernelGrid(const phd::svm::KernelTypes kernelType,
                                                                                const platform::Subtree& config) const
{
	switch (kernelType)
	{
	case phd::svm::KernelTypes::Rbf:
		return std::make_unique<RbfKernel>(RbfKernel(
			parseGridParameters("cGrid", config),
			parseGridParameters("gammaGrid", config), false));
	case phd::svm::KernelTypes::Linear:
		return std::make_unique<LinearKernel>(LinearKernel(
			parseGridParameters("cGrid", config), false));
	case phd::svm::KernelTypes::Poly:
		return std::make_unique<PolyKernel>(PolyKernel(
			parseGridParameters("cGrid", config),
			parseGridParameters("degreeGrid", config),
			false));
	case phd::svm::KernelTypes::RBF_POLY_GLOBAL:
		return std::make_unique<RbfPolyGlobalKernel>(RbfPolyGlobalKernel(
			parseGridParameters("cGrid", config),
			parseGridParameters("gammaGrid", config),
			parseGridParameters("degreeGrid", config),
			parseGridParameters("tGrid", config),
			false));
	default:
		throw GridSearchUnsupportedKernelTypeException(kernelType);
	}
}

void GridSearchConfiguration::validateGridName(const std::string& gridName)
{
	static const std::vector<std::string> paramsNames = {"cGrid", "gammaGrid", "pGrid", "nuGrid", "coefGrid", "degreeGrid", "tGrid"};
	if (std::find(paramsNames.begin(), paramsNames.end(), gridName) == paramsNames.end())
	{
		throw UnsupportedGridException(gridName);
	}
}
} // namespace svmComponents
