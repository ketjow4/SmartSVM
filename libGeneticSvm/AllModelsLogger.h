#pragma once

#include <filesystem>
//#include "libFileSystem/FileSystemDefinitions.h"
#include "libDataset/Dataset.h"
#include "libGeneticSvm/IDatasetLoader.h"
#include "libStrategies/DiskFile.h"
#include "libSvmStrategies/NormalizeStrategy.h"
//#include "libStrategies/TabularDataProviderStrategy.h"
#include "libGeneticComponents/Population.h"
#include "Timer.h"
#include <fstream>
#include <SvmLib/ISvm.h>

#include "libPlatform/TimeUtils.h"
#include "WorkflowUtils.h"

namespace genetic
{
class AllModelsLogger
{
public:
    AllModelsLogger(unsigned int number_of_run, std::string& outputPath, genetic::IDatasetLoader& loadingWorkflow)
        : m_numberOfRun(number_of_run)
        , m_outputPath(outputPath + "\\" + std::to_string(number_of_run))
        , m_loadingWorkflow(loadingWorkflow)
    {
        std::filesystem::create_directories(m_outputPath);
    }

    AllModelsLogger(const AllModelsLogger& other) = default;

    template <typename chromosome, typename ... Args>
    void log(const geneticComponents::Population<chromosome>& population,
             const geneticComponents::Population<chromosome>& populationWithTestDataFitness,
             Timer& timer,
             const std::string& algorithmName,
             unsigned int generationNumber,
             Args&&... additionalParameters);

	void save(const std::filesystem::path& outputPath)
	{
		//logger::LogFrontend logger;
		try
		{
			filesystem::DiskFile resultFile(outputPath, "a");
			for (const auto& logMessage : m_populationLogs)
			{
				resultFile.write(gsl::span<const unsigned char>(reinterpret_cast<const unsigned char*>(logMessage.c_str()), logMessage.length()));
			}
		}
		catch (const std::exception& /*exception*/)
		{
			//logger.LOG(logger::LogLevel::Error, exception.what());
		}
        m_populationLogs.clear();
	}

    unsigned int getNumberOfRun()
	{
        return m_numberOfRun;
	}

private:
    template <typename chromosome, typename ... Args>
    void logPopulationText(const geneticComponents::Population<chromosome>& population,
                           const geneticComponents::Population<chromosome>& populationWithTestDataFitness,
                           Timer& timer,
                           const std::string& algorithmName,
                           unsigned int generationNumber,
                           Args&&... additionalParameters);


    void saveSvmModel(std::filesystem::path outputPath,  int generation, int i, phd::svm::ISvm& resultModel, const std::string& configFile)
    {
        resultModel.save(outputPath.string() +
						 timeUtils::getTimestamp() + "_" +
                         configFile + "_" +
                         std::to_string(i) + "fold_" +
                         std::to_string(generation) + "_svmModel.xml");
    }

    void saveSvmResultsToFile(std::filesystem::path outputPath, int generation, int i, phd::svm::ISvm& resultModel, const std::string& configFile)
    {
        std::ofstream outputFile{ outputPath.string() +
            timeUtils::getTimestamp() + "_" + configFile + "_" + std::to_string(i) + "fold_" +
            std::to_string(generation) + "_svmModel.SvmResults", std::ios_base::out };

        if (outputFile.is_open())
        {
            auto featureSet = resultModel.getFeatureSet();
            auto featureChromosome = svmComponents::SvmFeatureSetMemeticChromosome(std::move(featureSet));
            auto training = featureChromosome.convertChromosome(m_loadingWorkflow.getTraningSet());
            auto validation = featureChromosome.convertChromosome(m_loadingWorkflow.getValidationSet());
            auto test = featureChromosome.convertChromosome(m_loadingWorkflow.getTestSet());

            outputFile << "#Training set" << "\n";

            for (auto& vec : training.getSamples())
            {
                outputFile << resultModel.classifyHyperplaneDistance(vec) << "\n";
            }

            outputFile << "#Validation set" << "\n";
            for (auto& vec : validation.getSamples())
            {
                outputFile << resultModel.classifyHyperplaneDistance(vec) << "\n";
            }

            outputFile << "#Test set" << "\n";
            for (auto& vec : test.getSamples())
            {
                outputFile << resultModel.classifyHyperplaneDistance(vec) << "\n";
            }
        }

        outputFile.close();
    }


    std::vector<std::string> m_populationLogs;
    unsigned int m_numberOfRun;
    std::string m_outputPath;
    genetic::IDatasetLoader& m_loadingWorkflow;
};

template <class chromosome>
void logKernelParameters(chromosome individual, std::string& logInfo)
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
        break;
    default:
        logInfo += "Custom kernel\t";
        //throw svmComponents::UnsupportedKernelTypeException(kernelType);
    }
}

template <typename ... Args>
std::string logAdditionalParameters(Args&&... additionalParameters)
{
    std::stringstream stream;
    stream << std::setprecision(6) << std::fixed;
    // @wdudzik in C++17 this can be changed to fold expression
    using expander = int[];
    (void)expander {
        0, ((void)(stream << additionalParameters << '\t'), 0) ...
    };

    return stream.str();
}

template <typename chromosome, typename ... Args>
void AllModelsLogger::log(const geneticComponents::Population<chromosome>& population,
                          const geneticComponents::Population<chromosome>& populationWithTestDataFitness,
                          Timer& timer, 
                          const std::string& algorithmName,
                          unsigned generationNumber, Args&&... additionalParameters)
{
    timer.pause();
    logPopulationText(population, populationWithTestDataFitness, timer, algorithmName, generationNumber, std::forward<Args>(additionalParameters)...);

    //save into output folder with create new folder with number of run

	//create folder for generation
	//filesystem::FileSystem fs;
	//auto outputPath = m_outputPath / std::filesystem::path(std::to_string(generationNumber));
	//fs.createDirectories(outputPath);

	//int i = 0;
 //   for(auto& individual : population)
 //   {
 //       //@wdudzik saving all svms in xml format of opencv takes to much disk space
	//	//saveSvmModel(std::filesystem::path(outputPath.string() + "\\"), generationNumber, i, *individual.getClassifier(), "configFileNameHere");
	//	
 //       saveSvmResultsToFile(outputPath / "", generationNumber, i, *individual.getClassifier(), "configFileNameHere");
	//	++i;
 //   }
    timer.contine();

}

template <typename chromosome, typename ...Args>
void AllModelsLogger::logPopulationText(const geneticComponents::Population<chromosome>& population,
                                        const geneticComponents::Population<chromosome>& populationWithTestDataFitness,
                                        Timer& timer,
                                        const std::string& algorithmName,
                                        unsigned int generationNumber,
                                        Args&&... additionalParameters)
{
    auto constexpr estimatedLogSize = 200u;
    int i = 0;
    for(auto& individual : population)
    {
        //auto bestIndividual = population.getBestOne();
        auto bestIndividual = individual;

        std::string logInfo;
        logInfo.reserve(estimatedLogSize);
        logInfo += std::to_string(i) + "\t";
        logInfo += algorithmName;
        logInfo += "\t";
        logInfo += std::to_string(generationNumber).append("\t");
        logInfo += std::to_string(population.size()).append("\t");
        logInfo += std::to_string(timer.getTimeMiliseconds().count()).append("\t");
        logInfo += std::to_string(bestIndividual.getFitness()).append("\t");
        logInfo += std::to_string(population.getMeanFitness()).append("\t");
        logInfo += std::to_string(bestIndividual.getNumberOfSupportVectors()).append("\t");
        logInfo += std::to_string(svmComponents::SvmPopulationStatistics<chromosome>::getMeanNumberOfSupportVectors(population)).append("\t");
        logInfo += std::to_string(populationWithTestDataFitness[i].getFitness()).append("\t");
        logInfo += std::to_string(populationWithTestDataFitness.getMeanFitness()).append("\t");
        logKernelParameters(bestIndividual, logInfo);
        logInfo += std::to_string(svmComponents::SvmPopulationStatistics<chromosome>::calculateUniqueElements(population)).append("\t");
        logInfo += std::to_string(bestIndividual.getTime().count()).append("\t");
        logInfo += std::to_string(svmComponents::SvmPopulationStatistics<chromosome>::getMeanTimeOfClassification(population)).append("\t");
        logInfo += std::to_string(populationWithTestDataFitness[i].getTime().count()).append("\t");
        logInfo += std::to_string(svmComponents::SvmPopulationStatistics<chromosome>::getMeanTimeOfClassification(populationWithTestDataFitness)).append("\t");
        logInfo += logAdditionalParameters(std::forward<Args>(additionalParameters)...);
        logInfo.append("\n");

        m_populationLogs.emplace_back(logInfo);
        ++i;
    }
}
} // namespace genetic
