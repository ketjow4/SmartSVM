#pragma once

#include "libSvmComponents/GeneticAlgorithmsConfigs.h"
#include "libGeneticSvm/ISvmAlgorithm.h"
#include "libGeneticSvm/SvmWorkflowConfigStruct.h"
#include "libGeneticSvm/Timer.h"
#include "libGeneticSvm/IGeneticWorkflow.h"
#include "libGeneticSvm/GeneticWorkflowResultLogger.h"
#include "libGeneticSvm/LocalFileDatasetLoader.h"
#include "libStrategies/FileSinkStrategy.h"
#include "libSvmStrategies/CreateSvmVisualizationStrategy.h"
#include "libGeneticStrategies/CreatePopulationStrategy.h"
#include "libGeneticStrategies/SelectionStrategy.h"
#include "libGeneticStrategies/MutationStrategy.h"
#include "libGeneticStrategies/CrossoverStrategy.h"
#include "libGeneticStrategies/StopConditionStrategy.h"
#include "libSvmComponents/SvmValidationStrategy.h"
#include "libSvmStrategies/SvmTrainingStrategy.h"
#include "libSvmComponents/SvmCustomKernelChromosome.h"
#include "libGeneticStrategies/CrossoverParentSelectionStrategy.h"

namespace genetic
{
    class CustomKernelWorkflow : public ISvmAlgorithm
    {
    public:
        explicit CustomKernelWorkflow(const SvmWokrflowConfiguration& config,
                               svmComponents::CustomKernelEvolutionConfiguration algorithmConfig,
                               IDatasetLoader& workflow);

        std::shared_ptr<phd::svm::ISvm> run() override;
     

        void runGeneticAlgorithm();

       /* void runGeneticAlgorithm() override;
        void initialize() override;
        void setupKernelParameters(const svmComponents::SvmKernelChromosome& kernelParameters) override;
        GeneticWorkflowResultLogger& getResultLogger() override;

        svmComponents::SvmTrainingSetChromosome getBestChromosomeInGeneration() const override;
        geneticComponents::Population<svmComponents::SvmTrainingSetChromosome> getPopulation() const override;
        dataset::Dataset<std::vector<float>, float> getBestTrainingSet() const override;
        void setupFeaturesSet(const svmComponents::SvmFeatureSetChromosome& featureSetChromosome) override;*/

        //geneticComponents::Population<svmComponents::SvmTrainingSetChromosome> initNoEvaluate(int /*popSize*/) override { throw; }
        //void performGeneticOperations(geneticComponents::Population<svmComponents::SvmTrainingSetChromosome>& /*population*/) override {}

    private:
        using SvmCustomKernelChromosome = svmComponents::SvmCustomKernelChromosome;

		void customLogTrainingSetAndGammas(std::shared_ptr<phd::svm::ISvm> svm);

        void logResults(const geneticComponents::Population<SvmCustomKernelChromosome>& population,
                        const geneticComponents::Population<SvmCustomKernelChromosome>& testPopulation);
        void initializeGeneticAlgorithm();

		void regionBasedScores();

		void internalVisualization_Debugging(geneticComponents::Population<svmComponents::SvmCustomKernelChromosome> pop);

   
        svmComponents::CustomKernelEvolutionConfiguration m_algorithmConfig;

        svmStrategies::SvmTrainingStrategy<SvmCustomKernelChromosome> m_trainingSvmClassifierElement;
        svmStrategies::SvmValidationStrategy<SvmCustomKernelChromosome> m_validationElement;
        svmStrategies::SvmValidationStrategy<SvmCustomKernelChromosome> m_validationTestDataElement;
        geneticStrategies::StopConditionStrategy<SvmCustomKernelChromosome> m_stopConditionElement;
        geneticStrategies::CrossoverStrategy<SvmCustomKernelChromosome> m_crossoverElement;
        geneticStrategies::MutationStrategy<SvmCustomKernelChromosome> m_mutationElement;
        geneticStrategies::SelectionStrategy<SvmCustomKernelChromosome> m_selectionElement;
        geneticStrategies::CreatePopulationStrategy<SvmCustomKernelChromosome> m_createPopulationElement;
        strategies::FileSinkStrategy m_savePngElement;
        svmStrategies::CreateSvmVisualizationStrategy<SvmCustomKernelChromosome> m_createVisualizationElement;
        geneticStrategies::CrossoverParentSelectionStrategy<SvmCustomKernelChromosome> m_crossoverParentSelectionElement;


        dataset::Dataset<std::vector<float>, float> m_trainingSet2;
        dataset::Dataset<std::vector<float>, float> m_validationSet2;
        dataset::Dataset<std::vector<float>, float> m_testSet2;

        const dataset::Dataset<std::vector<float>, float>* m_trainingSet;
        const dataset::Dataset<std::vector<float>, float>* m_validationSet;
        const dataset::Dataset<std::vector<float>, float>* m_testSet;
        geneticComponents::Population<SvmCustomKernelChromosome> m_population;
        std::filesystem::path m_pngNameSource;

        //bool m_needRetrain;
        IDatasetLoader& m_loadingWorkflow;
        Timer m_timer;
        unsigned int m_generationNumber;

        const SvmWokrflowConfiguration m_config;
        //logger::LogFrontend m_logger;
        GeneticWorkflowResultLogger m_resultLogger;
		GeneticWorkflowResultLogger m_region1Logger;
		GeneticWorkflowResultLogger m_region2Logger;

        static constexpr const char* m_algorithmName = "Custom Kernel ";
    };
} // namespace genetic
