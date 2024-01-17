
#pragma once

#include <libGeneticSvm/SvmAlgorithmFactory.h>

namespace strategies
{
class SvmTrainingStrategy
{
public:
    SvmTrainingStrategy() = default;

    std::string getDescription() const;
    std::shared_ptr<phd::svm::ISvm> launch(platform::Subtree& config);
   
};
} // namespace strategies