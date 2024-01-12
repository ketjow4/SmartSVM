

#include "StopConditionFactory.h"
#include "SelectionFactory.h"
#include "CrossoverSelectionFactory.h"
#include "BinaryCrossoverFactory.h"
#include "BinaryMutationFactory.h"
#include "BinaryGenerationFactory.h"

namespace geneticComponents
{
const std::unordered_map<std::string, StopCondition> StopConditionFactory::m_stopConditionTranslations =
{
    {"MeanFitness", StopCondition::MeanFitness},
    {"BestFitness", StopCondition::BestFitness}
};

const std::unordered_map<std::string, Selection> SelectionFactory::m_selectionTranslations =
{
    {"TruncationSelection", Selection::TruncationSelection},
    {"ConstatntTruncationSelection", Selection::ConstatntTruncationSelection}
};

const std::unordered_map<std::string, CrossoverSelection> CrossoverSelectionFactory::m_crossoverSelectionTranslations =
{
    {"HighLowFit", CrossoverSelection::HighLowFit},
    {"LocalGlobalSelection", CrossoverSelection::LocalGlobalSelection}
};

const std::unordered_map<std::string, BinaryCrossover> BinaryCrossoverFactory::m_crossoverTranslations =
{
    {"OnePoint", BinaryCrossover::OnePoint},
    {"FeaturesSelectionOnePoint", BinaryCrossover::FeaturesSelectionOnePoint}
};

const std::unordered_map<std::string, BinaryMutation> BinaryMutationFactory::m_mutationTranslations =
{
    {"BitFlip", BinaryMutation::BitFlip},
    {"FeaturesSelectionBitFlip", BinaryMutation::FeaturesSelectionBitFlip}
};

const std::unordered_map<std::string, BinaryGeneration> BinaryGenerationFactory::m_generationTranslations =
{
    {"Random", BinaryGeneration::Random},
    {"FeaturesSelectionRandom", BinaryGeneration::FeaturesSelectionRandom}
};
} // namespace geneticComponents
