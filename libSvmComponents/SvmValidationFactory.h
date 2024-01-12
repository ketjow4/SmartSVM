#pragma once

#include <memory>
#include <unordered_map>
#include "libPlatform/Subtree.h"
#include "libSvmComponents/SvmTrainingSetChromosome.h"
#include "libSvmComponents/SvmValidationStrategy.h"

#include "SvmComponentsExceptions.h"
#include "libSvmComponents/SvmSubsetValidation.h"
#include "ValidationSelectionMethodFactory.h"
#include "libPlatform/EnumStringConversions.h"

namespace svmComponents
{
enum class SvmValidation
{
	Unknown,
	Regular,
	Subset
};


class SvmValidationFactory
{
public:

	template <typename chromosome>
	static std::unique_ptr<svmStrategies::IValidationStrategy<chromosome>> create(const platform::Subtree& config,
	                                                                              const ISvmMetricsCalculator& metricCalculator);
private:
	const static std::unordered_map<std::string, SvmValidation> m_translations;
};

template <typename chromosome>
std::unique_ptr<svmStrategies::IValidationStrategy<chromosome>> SvmValidationFactory::create(const platform::Subtree& config,
	const ISvmMetricsCalculator& metricCalculator)
{
	auto name = config.getValue<std::string>("Validation.Name");

	switch (platform::stringToEnum(name, m_translations))
	{
	case SvmValidation::Regular:
	{
		return std::make_unique<svmStrategies::SvmValidationStrategy<chromosome>>(metricCalculator, false);
	}
	case SvmValidation::Subset:
	{
		auto subsetSelectionMethod = ValidationSelectionMethodFactory<chromosome>::create(config);
		return std::make_unique<svmStrategies::SvmSubsetValidation<chromosome>>(metricCalculator, subsetSelectionMethod);
	}
	default:
		//return std::make_unique<svmStrategies::SvmValidationStrategy<chromosome>>(metricCalculator);
		throw UnknownEnumType(name, typeid(svmStrategies::IValidationStrategy<SvmTrainingSetChromosome>).name());
	}
}
} // namespace svmComponents
