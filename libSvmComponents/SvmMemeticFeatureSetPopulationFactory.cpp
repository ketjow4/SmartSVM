#include "libPlatform/EnumStringConversions.h"
#include "libRandom/RandomNumberGeneratorFactory.h"
#include "SvmMemeticFeatureSetPopulationFactory.h"
#include "SvmUtils.h"
#include "RandomMemeticFeaturesGeneration.h"
#include "MemeticMutialInfoRoulleteWheelGeneration.h"

namespace svmComponents
{
const std::unordered_map<std::string, SvmMemeticFeatureSetGeneration> SvmMemeticFeatureSetPopulationFactory::m_translationsGeneration =
{
    {"Random", SvmMemeticFeatureSetGeneration::Random},
    {"MutualInfo", SvmMemeticFeatureSetGeneration::MutualInfo}
};

PopulationGeneration<SvmFeatureSetMemeticChromosome> SvmMemeticFeatureSetPopulationFactory::create(const platform::Subtree& config,
                                                                                                   const dataset::Dataset<std::vector<float>, float>&
                                                                                                   trainingSet,
                                                                                                   std::string trainingDataPath,
                                                                                                   std::string outputPath)
{
    auto name = config.getValue<std::string>("Generation.Name");

    switch (platform::stringToEnum(name, m_translationsGeneration))
    {
    case SvmMemeticFeatureSetGeneration::Random:
    {
        auto numberOfClassExamples = config.getValue<unsigned int>("NumberOfClassExamples");
        return std::make_unique<RandomMemeticFeaturesGeneration>(trainingSet,
                                                                 my_random::RandomNumberGeneratorFactory::create(config),
                                                                 numberOfClassExamples);
    }
    case SvmMemeticFeatureSetGeneration::MutualInfo:
    {
        auto numberOfClassExamples = config.getValue<unsigned int>("NumberOfClassExamples");
        return std::make_unique<MemeticMutialInfoRoulleteWheelGeneration>(trainingSet,
                                                                          my_random::RandomNumberGeneratorFactory::create(config),
                                                                          numberOfClassExamples,
                                                                          trainingDataPath,
															              outputPath);
    }
    default:
        throw UnknownEnumType(name, typeid(SvmMemeticFeatureSetGeneration).name());
    }
}
} // namespace svmComponents
