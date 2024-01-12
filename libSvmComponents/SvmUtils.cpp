
#include "SvmUtils.h"
#include "SvmComponentsExceptions.h"
#include "SvmConfigStructures.h"

namespace svmComponents { namespace svmUtils
{
unsigned int getNumberOfKernelParameters(phd::svm::KernelTypes kernelType, bool isRegression)
{
    switch (kernelType)
    {
    case phd::svm::KernelTypes::Rbf:
    {
        if(isRegression)
        {
            return 3;
        }
        return 2;
    }
    case phd::svm::KernelTypes::Linear:
    {
        if (isRegression)
        {
            return 2;
        }
        return 1;
    }
    case phd::svm::KernelTypes::Poly:
    {
        if (isRegression)
        {
            return 4;
        }
        return 3;
    }
	case phd::svm::KernelTypes::Sigmoid:
    {
        if (isRegression)
        {
            return 4;
        }
        return 3;
    }
    	
    default:
        throw UnsupportedKernelTypeException(kernelType);
    }
}

void setupSvmParameters(phd::svm::ISvm& svm, const SvmKernelChromosome& chromosome)
{
    if(chromosome.isRegression())
    {
        svm.setType(phd::svm::SvmTypes::EpsSvr);
        svm.setKernel(chromosome.getKernelType());
        switch (chromosome.getKernelType())
        {
        case phd::svm::KernelTypes::Rbf:
        {
            svm.setC(chromosome[0]);
            svm.setGamma(chromosome[1]);
            svm.setP(chromosome[2]);
            break;
        }
		case phd::svm::KernelTypes::Linear:
        {
			svm.setC(chromosome[0]);
			svm.setP(chromosome[1]);
			break;
        }
		case phd::svm::KernelTypes::Poly:
		{ 
        	svm.setC(chromosome[0]);
			svm.setDegree(chromosome[1]);
			svm.setP(chromosome[2]);
			break;
		}
        default:
            throw UnsupportedKernelTypeException(chromosome.getKernelType());
        }
    }
    else
    {
        svm.setType(phd::svm::SvmTypes::CSvc);
        svm.setKernel(chromosome.getKernelType());
        switch (chromosome.getKernelType())
        {
        case phd::svm::KernelTypes::Rbf:
		{
            svm.setC(chromosome[0]);
            svm.setGamma(chromosome[1]);
            break;
		}
		case phd::svm::KernelTypes::Linear:
		{
			svm.setC(chromosome[0]);
			break;
		}
		case phd::svm::KernelTypes::Poly:
		{
			svm.setC(chromosome[0]);
			svm.setDegree(chromosome[1]);
            svm.setCoef0(chromosome[2]);
            svm.setGamma(1); //there is possibility for setting it and scaling dot product, 1 is a neutral value
			break;
		}
        case phd::svm::KernelTypes::Sigmoid:
        {
            svm.setC(chromosome[0]);
            svm.setCoef0(chromosome[1]);
            svm.setGamma(chromosome[2]);
            break;
        }
        case phd::svm::KernelTypes::RBF_POLY_GLOBAL:
        {
            svm.setC(chromosome[0]);
            svm.setGamma(chromosome[1]);
            svm.setT(chromosome[2]);
            svm.setDegree(chromosome[3]);
            break;
        }
        default:
            throw UnsupportedKernelTypeException(chromosome.getKernelType());
        }
    }
   
}

std::uniform_real_distribution<double> getRange(const std::string& variablePath, const platform::Subtree& config)
{
    auto min = config.getValue<double>(variablePath + ".Min");
    auto max = config.getValue<double>(variablePath + ".Max");
    return std::uniform_real_distribution<double>(min, max);
}

std::vector<unsigned> countLabels(unsigned numberOfClasses, const std::vector<DatasetVector>& dataset)
{
    std::vector<unsigned int> labelsCount(numberOfClasses);

    std::for_each(dataset.begin(), dataset.end(),
                  [&labelsCount](const auto& dataVector)
                  {
                      ++labelsCount[static_cast<int>(dataVector.classValue)];
                  });
    return labelsCount;
}


std::vector<unsigned> countLabels(unsigned numberOfClasses, const std::vector<Gene>& dataset)
{
	std::vector<unsigned int> labelsCount(numberOfClasses);

	std::for_each(dataset.begin(), dataset.end(),
		[&labelsCount](const auto& dataVector)
	{
		++labelsCount[static_cast<int>(dataVector.classValue)];
	});
	return labelsCount;
}

void setupSvmTerminationCriteria(phd::svm::ISvm& svm, const SvmAlgorithmConfiguration& config)
{
    if (config.m_useSvmIteration)
    {
        svm.setTerminationCriteria(cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER,
                                                    config.m_svmIterationNumber,
                                                    config.m_svmEpsilon));
    }
    else
    {
        constexpr int numberOfIterations = 0;
        svm.setTerminationCriteria(cv::TermCriteria(cv::TermCriteria::EPS, numberOfIterations, config.m_svmEpsilon));
    }
}


std::vector<unsigned int> countLabels(unsigned int numberOfClasses,
	const dataset::Dataset<std::vector<float>, float>& dataset)
{
	std::vector<unsigned int> labelsCount(numberOfClasses);
	auto targets = dataset.getLabels();
	std::for_each(targets.begin(), targets.end(),
		[&labelsCount](const auto& label)
	{
		++labelsCount[static_cast<int>(label)];
	});
	return labelsCount;
}

//@wdudzik calcualtion as in numpy.var function with default parameters
double variance(const dataset::Dataset<std::vector<float>, float>& dataset)
{
    auto samples = dataset.getSamples();

    auto mean = 0.0;

    for (auto& sample : samples)
    {
        auto sumOfElems = std::accumulate(sample.begin(), sample.end(), 0.0);
        mean += sumOfElems;
    }
    auto size = samples.size() * samples[0].size(); //calcualte as flattened array
    mean = mean / size;

    auto variance = 0.0;

    for (auto& sample : samples)
    {
        for (auto value : sample)
        {
            variance += std::pow(std::fabs(value - mean), 2);
        }
    }
    variance = variance / size;

    return variance;
}


}} // namespace svmComponents::svmUtils
