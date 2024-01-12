

#pragma once

#include <memory>
#include "libPlatform/Subtree.h"
#include "libPlatform/EnumStringConversions.h"
#include "LibGeneticComponents/GeneticExceptions.h"
#include "LibGeneticComponents/ISelectionOperator.h"
#include "LibGeneticComponents/TruncationSelection.h"

namespace geneticComponents
{
enum class Selection
{
    Unknown,
    TruncationSelection,
	ConstatntTruncationSelection
};

class SelectionFactory
{
public:
    template<typename chromosome>
    static std::unique_ptr<ISelectionOperator<chromosome>> create(const platform::Subtree& config);

private:
    const static std::unordered_map<std::string, Selection> m_selectionTranslations;
};

template <typename chromosome>
std::unique_ptr<ISelectionOperator<chromosome>> SelectionFactory::create(const platform::Subtree& config)
{
    auto name = config.getValue<std::string>("SelectionOperator.Name");

    switch (platform::stringToEnum(name, m_selectionTranslations))
    {
    case Selection::TruncationSelection:
    {
        auto truncationCoefficient = config.getValue<double>("SelectionOperator.TruncationSelection.TruncationCoefficient");
        return std::make_unique<TruncationSelection<chromosome>>(platform::Percent(truncationCoefficient));
    }
    case Selection::ConstatntTruncationSelection:
    {
        int popSize = static_cast<int>(config.getValue<unsigned int>("PopulationSize"));
        return std::make_unique<ConstantTruncationSelection<chromosome>>(popSize);
    }
    default:
        throw UnknownEnumType(name, typeid(Selection).name());
    } 
}
} // namespace geneticComponents