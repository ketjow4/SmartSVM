
#include <algorithm>
#include "CompensationInformation.h"

namespace svmComponents
{
CompensationInformation::CompensationInformation(std::unique_ptr<my_random::IRandomNumberGenerator> randomNumberGenerator,
                                                 unsigned int numberOfClasses)
    : m_rngEngine(std::move(randomNumberGenerator))
    , m_numberOfClasses(numberOfClasses)
{
    if(m_rngEngine == nullptr)
    {
        throw RandomNumberGeneratorNullPointer();
    }
}

std::vector<unsigned int> CompensationInformation::generate(const std::vector<geneticComponents::Parents<SvmTrainingSetChromosome>>& parents,
                                                            unsigned int numberOfClassExamples) const
{
    std::vector<unsigned int> compensationInfo(parents.size());
    
    std::transform(parents.begin(), parents.end(), compensationInfo.begin(), [&,this](const auto& parentsPair)
    {
        auto parentsMax = static_cast<int>(std::max(parentsPair.first.getDataset().size(), parentsPair.second.getDataset().size()));

        auto min = static_cast<int>(std::min(parentsMax, static_cast<int>(m_numberOfClasses * numberOfClassExamples)));
        auto max = static_cast<int>(std::max(parentsMax, static_cast<int>(m_numberOfClasses * numberOfClassExamples)));
    	
        std::uniform_int_distribution<int> sizeOfChild(min, max);
        auto newSize = m_rngEngine->getRandom(sizeOfChild);

        if (newSize - parentsMax < 0) //due to changes in K for very small datasets this has to works like this (e.g. K=8 by default for small datasets it can be selected as 4 and here we have problem)
            return 0u;
    	
        return static_cast<unsigned int>(newSize - parentsMax);
    });
    return compensationInfo;
}
} // namespace svmComponents
