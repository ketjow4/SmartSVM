
#pragma once

#include "libGeneticComponents/Population.h"
#include "SvmKernelChromosome.h"

namespace svmComponents
{
class BaseSvmChromosome;


class ParamGrid {
public:
    ParamGrid(double min_val, double max_val, double logStep)
        : minVal(min_val), maxVal(max_val), logStep(logStep) {}

    std::vector<double> generateValues() const {
        std::vector<double> values;
        for (double current_val = minVal; current_val <= maxVal; current_val *= logStep) 
        {
            values.push_back(current_val);
        }
        return values;
    }

//parametersGrid.maxVal / parametersGrid.minVal) / log(parametersGrid.logStep))
    double minVal;
    double maxVal;
    double logStep;
};


class BaseKernelGridSearch
{
public:
    virtual ~BaseKernelGridSearch() = default;

    // @wdudzik for our grid search 
    virtual void calculateGrids(const BaseSvmChromosome& individual) = 0;
    virtual geneticComponents::Population<SvmKernelChromosome> createGridSearchPopulation() = 0;

    // @wdudzik for OpenCV grid search 
    // virtual void calculateGrids() = 0;
    // virtual std::string logSvmParameters() = 0;
    // virtual void performGridSearch(cv::Ptr<cv::ml::TrainData> trainingSet, unsigned int numberOfFolds) = 0;
    // void setSvm(cv::Ptr<cv::ml::SVM> svm);

protected:
    ParamGrid calculateNewGrid(const ParamGrid& parametersGrid, double gridParemeter) const;
    static unsigned int calculateNumberOfSteps(const ParamGrid& parametersGrid);

    //cv::Ptr<cv::ml::SVM> m_svm;
};

} // namespace svmComponents
