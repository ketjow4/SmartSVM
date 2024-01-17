
#pragma once

#include <string>
#include <iomanip>
#include <vector>
#include <mutex>
#include <filesystem>
#include "libGeneticComponents/Population.h"
#include "libSvmComponents/SvmPopulationStatistics.h"
#include "libSvmComponents/SvmComponentsExceptions.h"
#include "libGeneticSvm/Timer.h"

namespace genetic
{
inline std::string header()
{
    auto constexpr estimatedLogSize = 200u;

    static bool isRun = false;

    std::string logInfo;
    logInfo.reserve(estimatedLogSize);
    logInfo += "Name";
    logInfo += "\t";
    logInfo += "genNum\t";
    logInfo += "popSize\t";
    logInfo += "time\t";//std::to_string(timer.getTimeMiliseconds().count()).append("\t");
    logInfo += "V\t";//std::to_string(bestIndividual.getFitness()).append("\t");
    logInfo += "V sr\t";//std::to_string(population.getMeanFitness()).append("\t");
    logInfo += "S\t";//std::to_string(bestIndividual.getNumberOfSupportVectors()).append("\t");
    logInfo += "S sr\t";//std::to_string(svmComponents::SvmPopulationStatistics<chromosome>::getMeanNumberOfSupportVectors(population)).append("\t");
    logInfo += "Psi\t";//std::to_string(populationWithTestDataFitness[population.getBestIndividualIndex()].getFitness()).append("\t");
    logInfo += "Psi sr\t";//std::to_string(populationWithTestDataFitness.getMeanFitness()).append("\t");
    logInfo += "Gamma\tC\t";//logKernelParameters(bestIndividual, logInfo);
    //logInfo += //std::to_string(svmComponents::SvmPopulationStatistics<chromosome>::calculateUniqueElements(population)).append("\t");
    //logInfo += //std::to_string(bestIndividual.getTime().count()).append("\t");
    //logInfo += //std::to_string(svmComponents::SvmPopulationStatistics<chromosome>::getMeanTimeOfClassification(population)).append("\t");
    //logInfo += //std::to_string(populationWithTestDataFitness[population.getBestIndividualIndex()].getTime().count()).append("\t");
    //logInfo += //std::to_string(svmComponents::SvmPopulationStatistics<chromosome>::getMeanTimeOfClassification(populationWithTestDataFitness)).append("\t");
    //logInfo += //logAdditionalParameters(std::forward<Args>(additionalParameters)...);
    logInfo.append("\n");

    if(!isRun)
    {
        isRun = true;
        return logInfo;
        
    }
    return std::string();
}

class GeneticWorkflowResultLogger
{
public:
    template <class chromosome, typename ... Args>
    void createLogEntry(const geneticComponents::Population<chromosome>& population,
                        const geneticComponents::Population<chromosome>& populationWithTestDataFitness,
                        const Timer& timer,
                        const std::string& algorithmName,
                        unsigned int generationNumber,
				        double accuracy,
				        std::size_t featureNumber,
                        Args&&... additionalParameters);

    void logToFile(const std::filesystem::path& outputPath);

    void logToConsole();

    void setEntries(std::vector<std::string>& entires);

	void customLogEntry(const std::string& logEntry)
	{
		m_resultEntries.emplace_back(logEntry);
	}

    const std::vector<std::string>& getLogEntries() const;

    void clearLog();

private:
    template <class chromosome>
    static void logKernelParameters(chromosome individual, std::string& logInfo);

    template< typename ... Args >
    static std::string logAdditionalParameters(Args&& ... args);

    std::vector<std::string> m_resultEntries;
    std::once_flag m_once;
};

template <class chromosome, typename ... Args>
void GeneticWorkflowResultLogger::createLogEntry(const geneticComponents::Population<chromosome>& population,
                                                 const geneticComponents::Population<chromosome>& populationWithTestDataFitness,
                                                 const Timer& timer,
                                                 const std::string& algorithmName,
                                                 unsigned int generationNumber,
												 double accuracy,
											     std::size_t featureNumber,
                                                 Args&&... additionalParameters)
{
    std::call_once(m_once, [&]()
                   {
                       auto headerTxt = header();
                       m_resultEntries.emplace_back(headerTxt);
                   });

    
    auto bestIndividual = population.getBestOne();

    auto constexpr estimatedLogSize = 200u;

    std::string logInfo;
    logInfo.reserve(estimatedLogSize);
    logInfo += algorithmName;
    logInfo += "\t";
    logInfo += std::to_string(generationNumber).append("\t");
    logInfo += std::to_string(population.size()).append("\t");
    logInfo += std::to_string(timer.getTimeMiliseconds().count()).append("\t");
    logInfo += std::to_string(bestIndividual.getFitness()).append("\t");
    logInfo += std::to_string(population.getMeanFitness()).append("\t");
    logInfo += std::to_string(bestIndividual.getNumberOfSupportVectors()).append("\t");
    logInfo += std::to_string(svmComponents::SvmPopulationStatistics<chromosome>::getMeanNumberOfSupportVectors(population)).append("\t");
    logInfo += std::to_string(populationWithTestDataFitness[population.getBestIndividualIndex()].getFitness()).append("\t");
    logInfo += std::to_string(populationWithTestDataFitness.getMeanFitness()).append("\t");
    logKernelParameters(bestIndividual, logInfo);
    logInfo += std::to_string(svmComponents::SvmPopulationStatistics<chromosome>::calculateUniqueElements(population)).append("\t");
    logInfo += std::to_string(bestIndividual.getTime().count()).append("\t");
    logInfo += std::to_string(svmComponents::SvmPopulationStatistics<chromosome>::getMeanTimeOfClassification(population)).append("\t");
    logInfo += std::to_string(populationWithTestDataFitness[population.getBestIndividualIndex()].getTime().count()).append("\t");
    logInfo += std::to_string(svmComponents::SvmPopulationStatistics<chromosome>::getMeanTimeOfClassification(populationWithTestDataFitness)).append("\t");
    logInfo += std::to_string(accuracy).append("\t");
    logInfo += std::to_string(featureNumber).append("\t");
    logInfo += logAdditionalParameters(std::forward<Args>(additionalParameters)...);
    logInfo.append("\n");

    m_resultEntries.emplace_back(logInfo);
}

template <class chromosome>
void GeneticWorkflowResultLogger::logKernelParameters(chromosome individual, std::string& logInfo)
{
    static_assert(std::is_base_of<svmComponents::BaseSvmChromosome, chromosome>::value, "Cannot do log parameters for svm for non-svm chromosome");

    if (!individual.getClassifier()->isTrained())
    {
        throw svmComponents::UntrainedSvmClassifierException();
    }

    auto kernelType = (individual.getClassifier()->getKernelType());

    switch (kernelType)
    {
    case phd::svm::KernelTypes::Rbf:
        logInfo += std::to_string(individual.getClassifier()->getGamma()).append("\t");
        logInfo += std::to_string(individual.getClassifier()->getC()).append("\t");
        //logInfo += std::to_string(individual.getClassifier()->getP()).append("\t");
        break;
	case phd::svm::KernelTypes::Linear:
		logInfo += std::to_string(individual.getClassifier()->getC()).append("\t");
		break;
    default:
        logInfo += "Custom kernel\t"; 
    }
}

template <typename ... Args>
std::string GeneticWorkflowResultLogger::logAdditionalParameters(Args&&... additionalParameters)
{
    std::stringstream stream;
    stream << std::setprecision(6) << std::fixed;
    // @wdudzik in C++17 this can be changed to fold expression
    using expander = int[];
    (void)expander{0, ((void)(stream << additionalParameters << '\t') , 0) ...};

    return stream.str();
}

inline const std::vector<std::string>& GeneticWorkflowResultLogger::getLogEntries() const
{
    return m_resultEntries;
}

inline void GeneticWorkflowResultLogger::clearLog()
{
    m_resultEntries.clear();
}
} // namespace genetic
