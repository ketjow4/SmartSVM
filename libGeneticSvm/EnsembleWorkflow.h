#pragma once

#include "libSvmComponents/GeneticAlgorithmsConfigs.h"
#include "libSvmComponents/SvmKernelChromosome.h"
#include "libGeneticSvm/ISvmAlgorithm.h"
#include "libGeneticSvm/SvmWorkflowConfigStruct.h"
#include "libGeneticSvm/Timer.h"
#include "libGeneticSvm/IGeneticWorkflow.h"
#include "libGeneticSvm/GeneticWorkflowResultLogger.h"
#include "libGeneticSvm/IDatasetLoader.h"

#include "libSvmComponents/SvmValidationStrategy.h"
#include "libGeneticStrategies/StopConditionStrategy.h"
#include "libGeneticStrategies/CrossoverStrategy.h"
#include "libGeneticStrategies/MutationStrategy.h"
#include "libGeneticStrategies/SelectionStrategy.h"
#include "libGeneticStrategies/CreatePopulationStrategy.h"
#include "libStrategies/FileSinkStrategy.h"
#include "libSvmStrategies/CreateSvmVisualizationStrategy.h"
#include "libSvmStrategies/SvmTrainingStrategy.h"
#include "libSvmStrategies/MemeticEducationStrategy.h"
#include "libSvmStrategies/CrossoverCompensationStrategy.h"
#include "libGeneticStrategies/CombinePopulationsStrategy.h"
#include "libSvmStrategies/UpdateSupportVectorPoolStrategy.h"
#include "libSvmStrategies/SuperIndividualCreationStrategy.h"
#include "libSvmStrategies/MemeticAdaptationStrategy.h"
#include "libSvmStrategies/CompensationInformationStrategy.h"
#include "libGeneticStrategies/CrossoverParentSelectionStrategy.h"
#include "AllModelsLogger.h"
#include "libGeneticSvm/CombinedAlgorithmsConfig.h"
#include "libPlatform/StringUtils.h"
#include "SvmLib/EnsembleSvm.h"
#include "libSvmComponents/ConfusionMatrixMetrics.h"

//#include "libSvmComponents/SvmMetricFactory.h"

namespace genetic
{
struct GeneticAlternatingEvolutionConfiguration;

class EnsembleWorkflow : public ISvmAlgorithm
    {
    public:
        explicit EnsembleWorkflow(const SvmWokrflowConfiguration& config,
            GeneticAlternatingEvolutionConfiguration algorithmConfig,
            IDatasetLoader& workflow);

        std::shared_ptr<phd::svm::ISvm> run() override;

    private:
        template<class  chromosome>
        void ensemble(geneticComponents::Population<chromosome>& pop);
	
        bool isFinished() const;

        template<class chromosome>
        void log(IGeneticWorkflow<chromosome>& workflow);

        template <class chromosome>
        void clearlog(IGeneticWorkflow<chromosome>& workflow);

        void logResultsEnsemble(phd::svm::EnsembleSvm& ensembleSvm);

        SvmWokrflowConfiguration m_config;
        TrainingSetOptimizationWorkflow m_trainingSetOptimization;
        KernelOptimizationWorkflow m_kernelOptimization;
        std::filesystem::path m_resultFilePath;
        GeneticAlternatingEvolutionConfiguration m_algorithmConfig;
        IDatasetLoader& m_loadingWorkflow;
        strategies::FileSinkStrategy m_savePngElement;
        std::filesystem::path m_outputPath;
        Timer m_timer;

        //logger::LogFrontend m_logger;
};

template <class chromosome>
void EnsembleWorkflow::log(IGeneticWorkflow<chromosome>& workflow)
{
    workflow.getResultLogger().logToFile(m_resultFilePath);
    workflow.getResultLogger().clearLog();
}

template <class chromosome>
void EnsembleWorkflow::clearlog(IGeneticWorkflow<chromosome>& workflow)
{
    workflow.getResultLogger().clearLog();
}


inline std::string getLastLine(std::filesystem::path& path)
{
    std::ifstream fin;
    fin.open(path.string(), std::ios_base::in);
    if (fin.is_open())
    {
        std::string lastline;
        while (fin >> std::ws && std::getline(fin, lastline)) // skip empty lines
            ;
        fin.close();
        return lastline;
    }
    throw std::exception(std::string("Cannot open file: " + path.string()).c_str());
}


inline void EnsembleWorkflow::logResultsEnsemble(phd::svm::EnsembleSvm& ensembleSvm)
{
    auto trainingSetSize = m_loadingWorkflow.getTraningSet().size();

    auto metric = svmComponents::SvmMetricFactory::create(svmComponents::svmMetricType::Auc);
    svmStrategies::SvmValidationStrategy< svmComponents::SvmKernelChromosome> validationStrategy(*metric.get(), false);
    svmStrategies::SvmValidationStrategy< svmComponents::SvmKernelChromosome> validationStrategyTest(*metric.get(), true);

    svmComponents::SvmKernelChromosome ch;
    ch.updateClassifier(std::make_shared<phd::svm::EnsembleSvm>(ensembleSvm));
    svmComponents::SvmKernelChromosome chTest;
    chTest.updateClassifier(std::make_shared<phd::svm::EnsembleSvm>(ensembleSvm));

    geneticComponents::Population<svmComponents::SvmKernelChromosome> pop(std::vector<svmComponents::SvmKernelChromosome>{ ch });
    geneticComponents::Population<svmComponents::SvmKernelChromosome> popTest(std::vector<svmComponents::SvmKernelChromosome>{ chTest });
	
    pop = validationStrategy.launch(pop, m_loadingWorkflow.getValidationSet());
    popTest = validationStrategyTest.launch(popTest, m_loadingWorkflow.getTestSet());

	
    auto bestOneConfustionMatrix = pop[0].getConfusionMatrix().value();
    auto featureNumber = m_loadingWorkflow.getValidationSet().getSamples()[0].size();

    auto& m_resultLogger = m_kernelOptimization->getResultLogger();
   
	
    m_resultLogger.createLogEntry(pop,
        popTest,
        m_timer,
        "Ensemble",
        0,
        svmComponents::Accuracy(bestOneConfustionMatrix),
        featureNumber,
        trainingSetSize,
        bestOneConfustionMatrix,
        popTest[0].getConfusionMatrix().value());


    m_resultLogger.logToFile(m_resultFilePath);
}

template<class  chromosome>
void EnsembleWorkflow::ensemble(geneticComponents::Population<chromosome>& pop)
{
    std::vector<phd::svm::libSvmImplementation*> svms;
    int i = 0;
    for (auto& p : pop)
    {
        //if (i < 5) //select only 5 best ones to create ensemble
        //{
        //    ++i;
        //    continue;
        //}
        svms.emplace_back(reinterpret_cast<phd::svm::libSvmImplementation*>(p.getClassifier().get()));
        ++i;
    }

    phd::svm::EnsembleSvm ensemble(svms);

    auto metric = svmComponents::SvmMetricFactory::create(svmComponents::svmMetricType::Auc);
    svmComponents::BaseSvmChromosome ch;
    ch.updateClassifier(std::make_shared<phd::svm::EnsembleSvm>(ensemble));
    auto fitness = metric->calculateMetric(ch, m_loadingWorkflow.getValidationSet(), false);

    if (fitness.m_fitness > pop.getBestOne().getFitness())
    {
        std::cout << "Ensemble is better :) \n";
        std::cout << "Ensemble " << fitness.m_fitness << " single best one: " << pop.getBestOne().getFitness() << "\n";
    }
    else
    {
        std::cout << "Ensemble " << fitness.m_fitness << " single best one: " << pop.getBestOne().getFitness() << "\n";
    }

    if(m_algorithmConfig.m_svmConfig.m_doVisualization)
    {
	    std::filesystem::path output;
	    svmComponents::SvmVisualization visualization2;
	    auto image2 = visualization2.createDetailedVisualization(ensemble, 500, 500, m_loadingWorkflow.getTraningSet(), m_loadingWorkflow.getTraningSet(), m_loadingWorkflow.getTestSet());
	    SvmWokrflowConfiguration config_copy2{ "", "", "", m_outputPath, "ensemble", "" };
	    setVisualizationFilenameAndFormat(svmComponents::imageFormat::png, output, config_copy2, 0);
	    m_savePngElement.launch(image2, output);
    }
    logResultsEnsemble(ensemble);
	
    //std::vector<std::string> a;
    //auto finalEntry = ::platform::stringUtils::splitString(getLastLine(m_resultFilePath), '\t');

	////name
 //   finalEntry[0] = "Ensemble";

 //   finalEntry[4] = fitness.m_fitness;
 //   finalEntry[5] = fitness.m_fitness;

 //   finalEntry[6] = ensemble.getNumberOfSupportVectors();
 //   finalEntry[7] = ensemble.getNumberOfSupportVectors();

 //   auto fitnessTest = metric->calculateMetric(ch, m_loadingWorkflow.getValidationSet());

 //   finalEntry[8] = fitnessTest.m_fitness;
 //   finalEntry[9] = fitnessTest.m_fitness;

 //   //matrix.truePositive() << "\t" << matrix.falsePositive() << "\t" << matrix.trueNegative() << "\t" << matrix.falseNegative();
	//
	//
 //   //finalEntry[3] = time;

 //   std::string s = std::accumulate(std::begin(finalEntry), std::end(finalEntry), std::string(),
 //       [](std::string &ss, std::string &s)
 //   {
 //       return ss.empty() ? s : ss + "\t" + s;
 //   });
 //   //s.append("\n");

 //   a.push_back(s);
 //   m_kernelOptimization->getResultLogger().setEntries(a);
 //   m_kernelOptimization->getResultLogger().logToFile(m_resultFilePath);


}
} // namespace genetic
