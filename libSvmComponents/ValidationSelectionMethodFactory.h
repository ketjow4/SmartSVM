#pragma once

#include <memory>
#include <unordered_map>
#include "libPlatform/Subtree.h"
#include "libSvmComponents/SvmSubsetValidation.h"
#include "libPlatform/EnumStringConversions.h"
#include "libRandom/RandomNumberGeneratorFactory.h"

namespace svmComponents
{
enum class ValidationSelectionMethod
{
	Unknown,
	Dummy,
	RandomSubsetPerIteration,
};

template<class chromosome>
class ValidationSelectionMethodFactory
{
public:
	static std::shared_ptr<svmStrategies::IValidationSubsetSelection<chromosome>> create(const platform::Subtree& config);
private:
	const static std::unordered_map<std::string, ValidationSelectionMethod> m_translations;
};

template <typename T> const std::unordered_map<std::string, ValidationSelectionMethod> ValidationSelectionMethodFactory<T>::m_translations =
{
	{"Dummy", ValidationSelectionMethod::Dummy},
	{"RandomSubsetPerIteration", ValidationSelectionMethod::RandomSubsetPerIteration},
};

template <class chromosome>
std::shared_ptr<svmStrategies::IValidationSubsetSelection<chromosome>> ValidationSelectionMethodFactory<chromosome>::create(const platform::Subtree& config)
{
	auto name = config.getValue<std::string>("Validation.Method");

	switch (platform::stringToEnum(name, m_translations))
	{
	case ValidationSelectionMethod::Dummy:
	{
		return std::make_shared<svmStrategies::DummySelection<chromosome>>();
	}
	case ValidationSelectionMethod::RandomSubsetPerIteration:
	{
		auto percentSize = platform::Percent(config.getValue<double>("Validation.RandomSubsetPercent"));
		return std::make_shared<svmStrategies::RandomSubsetPerIteration<chromosome>>(percentSize,
		                                                                             std::move(random::RandomNumberGeneratorFactory::create(config)));
	}
	default:
		throw UnknownEnumType(name, typeid(svmStrategies::IValidationStrategy<SvmTrainingSetChromosome>).name());
	}
}
} // namespace svmComponents
