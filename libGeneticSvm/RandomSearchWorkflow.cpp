#include "RandomSearchWorkflow.h"
#include "libSvmComponents/ConfusionMatrixMetrics.h"
#include "libSvmComponents/SvmKernelRandomGeneration.h"
#include "libRandom/MersenneTwister64Rng.h"
#include "libPlatform/StringUtils.h"

namespace genetic
{
RandomSearchWorkflow::RandomSearchWorkflow(const SvmWokrflowConfiguration& config, 
                                           RandomSearchWorkflowConfig algorithmConfig,
                                           IDatasetLoader& workflow)
    : m_resultFilePath(std::filesystem::path(config.outputFolderPath.string() + config.txtLogFilename))
    , m_algorithmConfig(algorithmConfig)
    , m_svmTraining(*m_algorithmConfig.m_svmTraining)
    , m_workflow(workflow)
    , m_config(config)
    , m_generationNumber(0)
    , m_trainingSet(workflow.getTraningSet())
    , m_validationSet(workflow.getValidationSet())
    , m_testSet(workflow.getTestSet())
    , m_validation(*algorithmConfig.m_svmConfig.m_estimationMethod, false)
    , m_validationTest(*algorithmConfig.m_svmConfig.m_estimationMethod, true)
{
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


std::filesystem::path get_all(const std::filesystem::path& root, const std::string& nameToFind)
{
    //filesystem::FileSystem fs;
    if (!std::filesystem::exists(root) || !std::filesystem::is_directory(root))
        return {};

    std::filesystem::recursive_directory_iterator it(root);
    std::filesystem::recursive_directory_iterator endit;

    while (it != endit)
    {
        if (std::filesystem::is_regular_file(*it) && it->path().string().find(nameToFind) != std::string::npos)
        {
            return it->path();
        }
        ++it;

    }
    return {};
}

struct EvoHelpResult
{
    EvoHelpResult(int tries,
                  int supportVectorNumber,
                  int featureNumber)
        : m_tries(tries), m_supportVectorNumber(supportVectorNumber), m_featureNumber(featureNumber)
    {}

    int m_tries;
    int m_supportVectorNumber;
    int m_featureNumber;
};

EvoHelpResult numberOfTries(std::string& outputPath)
{
    std::vector<std::string> allRunsResults;

    auto summaryFilePath = get_all(outputPath + "..\\SSVM", "summary");

    std::ifstream input{ summaryFilePath };

    std::string s;
    while (getline(input, s))
    {
        allRunsResults.emplace_back(s);
    }
    input.close();

    auto generations = 0;
    auto popSize = 0;
    auto supportVectorNumber = 0;
    auto featureNumber = 0;
    auto numberOfRunsPerFold = static_cast<int>(allRunsResults.size());
    
    for (auto i = 0; i < allRunsResults.size(); ++i)
    {
        constexpr auto generationColumn = 1;
        constexpr auto supportVectorColumn = 6;
        constexpr auto featureColumn = 18;   //18
        auto entry = ::platform::stringUtils::splitString(allRunsResults[i], '\t');
        generations += std::stoi(entry[generationColumn]);
        supportVectorNumber += std::stoi(entry[supportVectorColumn]);
        featureNumber += std::stoi(entry[featureColumn]);
        popSize = std::stoi(entry[2]);
    }
    auto tries = (generations / numberOfRunsPerFold) * popSize;
    featureNumber /= numberOfRunsPerFold;
    supportVectorNumber /= numberOfRunsPerFold;
    
    return EvoHelpResult{ tries, supportVectorNumber,featureNumber };
}


geneticComponents::Population<svmComponents::SvmTrainingSetChromosome> createPopulationTrainingSet(uint32_t populationSize,
                                                                                        dataset::Dataset<std::vector<float>, float>& m_trainingSet,
                                                                                        int m_numberOfClasses, bool isFeateureNumberRandom = true, unsigned int numberOfClassExamples = 0)
{
    if (populationSize == 0)
    {
        throw geneticComponents::PopulationIsEmptyException();
    }
    std::unique_ptr<my_random::IRandomNumberGenerator> m_rngEngine = std::make_unique<my_random::MersenneTwister64Rng>(static_cast<unsigned long long>(std::chrono::system_clock::now().time_since_epoch().count()));

    auto targets = m_trainingSet.getLabels();
    auto trainingSetID = std::uniform_int_distribution<int>(0, static_cast<int>(m_trainingSet.size() - 1));
    std::vector<svmComponents::SvmTrainingSetChromosome> population(populationSize);
    auto labelCount = countLabels(m_numberOfClasses, m_trainingSet);

    //std::generate(population.begin(), population.end(), [&]
    //              {
    //                  std::unordered_set<std::uint64_t> trainingSet;
    //                  std::vector<svmComponents::DatasetVector> chromosomeDataset;
    //                  unsigned int m_numberOfClassExamples = m_rngEngine->getRandom(trainingSetID) + 1; //we don't want 0 in this case and the max will be ok
    //                  if (std::any_of(labelCount.begin(), labelCount.end(), [&](const auto& labelCount) { return m_numberOfClassExamples > labelCount;  }))
    //                  {
    //                      m_numberOfClassExamples = *std::min_element(labelCount.begin(), labelCount.end());
    //                  }

    //                  chromosomeDataset.reserve(m_numberOfClassExamples * m_numberOfClasses);
    //                  std::vector<unsigned int> classCount(m_numberOfClasses, 0);
    //                  while (std::any_of(classCount.begin(), classCount.end(), [&](const auto& classIndicies) { return classIndicies != m_numberOfClassExamples;  }))
    //                  {
    //                      auto randomValue = m_rngEngine->getRandom(trainingSetID);
    //                      if (classCount[targets[randomValue]] < m_numberOfClassExamples &&    // less that desired number of class examples
    //                          trainingSet.emplace(static_cast<int>(randomValue)).second)       // is unique
    //                      {
    //                          chromosomeDataset.emplace_back(svmComponents::DatasetVector(randomValue, static_cast<std::uint8_t>(targets[randomValue])));
    //                          classCount[targets[randomValue]]++;
    //                      }
    //                  }
    //                  return svmComponents::SvmTrainingSetChromosome(std::move(chromosomeDataset));
    //              });


    //NO BALANCING 
    std::generate(population.begin(), population.end(), [&]
                  {
                      std::unordered_set<std::uint64_t> trainingSet;
                      std::vector<svmComponents::DatasetVector> chromosomeDataset;

                      unsigned int m_numberOfClassExamples;
                      if (isFeateureNumberRandom)
                          m_numberOfClassExamples = m_rngEngine->getRandom(trainingSetID) + 1; //we don't want 0 in this case and the max will be ok
                      else
                          m_numberOfClassExamples = numberOfClassExamples;

                      chromosomeDataset.reserve(m_numberOfClassExamples * m_numberOfClasses);
                      std::vector<unsigned int> classCount(m_numberOfClasses, 0);
                      while (std::any_of(classCount.begin(), classCount.end(), [&](const auto& classIndicies) { return classIndicies == 0;  }) || chromosomeDataset.size() < m_numberOfClassExamples)
                      {
                          auto randomValue = m_rngEngine->getRandom(trainingSetID);
                          if (trainingSet.emplace(static_cast<int>(randomValue)).second)       // is unique
                          {
                              if (std::any_of(classCount.begin(), classCount.end(), [&](const auto& classIndicies) { return classIndicies == 0;  }) &&
                                  chromosomeDataset.size() == m_numberOfClassExamples - 1
                                  && classCount[static_cast<int>(targets[randomValue])] != 0)
                              {
                                  trainingSet.erase(static_cast<int>(randomValue));
                                  continue;
                              }
                              chromosomeDataset.emplace_back(svmComponents::DatasetVector(randomValue, static_cast<std::uint8_t>(targets[randomValue])));
                              classCount[static_cast<int>(targets[randomValue])]++;
                          }
                      }
                      return svmComponents::SvmTrainingSetChromosome(std::move(chromosomeDataset));
                  });
    return geneticComponents::Population<svmComponents::SvmTrainingSetChromosome>(std::move(population));
}


geneticComponents::Population<svmComponents::SvmFeatureSetMemeticChromosome> createPopulationFeatures(uint32_t populationSize, 
                                                                               dataset::Dataset<std::vector<float>, float>& m_trainingSet, bool isFeateureNumberRandom = true ,unsigned int numberOfClassExamples = 0)
{
    if (populationSize == 0)
    {
        throw geneticComponents::PopulationIsEmptyException();
    }
    std::unique_ptr<my_random::IRandomNumberGenerator> m_rngEngine = std::make_unique<my_random::MersenneTwister64Rng>(static_cast<unsigned long long>(std::chrono::system_clock::now().time_since_epoch().count()));

    auto trainingSetID = std::uniform_int_distribution<int>(0, static_cast<int>(m_trainingSet.getSample(0).size() - 1));
    std::vector<svmComponents::SvmFeatureSetMemeticChromosome> population(populationSize);

    std::generate(population.begin(), population.end(), [&]
                  {
                      std::unordered_set<std::uint64_t> trainingSet;
                      std::vector<svmComponents::Feature> chromosomeDataset;
                      unsigned int m_numberOfClassExamples;
                      if (isFeateureNumberRandom)
                          m_numberOfClassExamples = m_rngEngine->getRandom(trainingSetID) + 1; //we don't want 0 in this case and the max will be ok
                      else
                          m_numberOfClassExamples = numberOfClassExamples;

                      chromosomeDataset.reserve(m_numberOfClassExamples);
                      std::vector<unsigned int> classCount(1, 0);
                      while (std::any_of(classCount.begin(), classCount.end(), [&](const auto& classIndicies) { return classIndicies != m_numberOfClassExamples;  }))
                      {
                          auto randomValue = m_rngEngine->getRandom(trainingSetID);
                          if (classCount[0] < m_numberOfClassExamples &&    // less that desired number of class examples
                              trainingSet.emplace(static_cast<int>(randomValue)).second)       // is unique
                          {
                              chromosomeDataset.emplace_back(svmComponents::Feature(randomValue));
                              classCount[0]++;
                          }
                      }
                      return svmComponents::SvmFeatureSetMemeticChromosome(std::move(chromosomeDataset));
                  });
    return geneticComponents::Population<svmComponents::SvmFeatureSetMemeticChromosome>(std::move(population));
}


geneticComponents::Population<svmComponents::SvmSimultaneousChromosome> RandomSearchWorkflow::initRandom(int numberOfTries)
{
    const auto maxTraningSetSize = m_trainingSet.size();
    const auto maxFeatureNumber = m_trainingSet.getSample(0).size();

    // "Min": "0.0001",
    // "Max": "1000.1"
    //get kernel
    std::unique_ptr<my_random::IRandomNumberGenerator> rnd = std::make_unique<my_random::MersenneTwister64Rng>(static_cast<unsigned long long>(std::chrono::system_clock::now().time_since_epoch().count()));
    svmComponents::SvmKernelRandomGeneration generation(std::uniform_real_distribution<double>(0.0001, 1000.1), phd::svm::KernelTypes::Rbf, std::move(rnd), false);

    auto kernelPop = generation.createPopulation(numberOfTries);

    //get training set
    constexpr auto numberOfClasses = 2;
    auto trainingPop = createPopulationTrainingSet(numberOfTries, m_trainingSet, numberOfClasses);

    //get feature set
    auto featureSetPopulation = createPopulationFeatures(numberOfTries, m_trainingSet);


    std::vector<svmComponents::SvmSimultaneousChromosome> vec;
    vec.reserve(numberOfTries);
    for (auto i = 0; i < numberOfTries; ++i)
    {
        svmComponents::SvmSimultaneousChromosome c{ kernelPop[i], trainingPop[i], featureSetPopulation[i] };
        vec.emplace_back(c);
    }
    geneticComponents::Population<svmComponents::SvmSimultaneousChromosome> pop{ vec };
    m_pop = pop;

    return m_pop;
}

std::shared_ptr<phd::svm::ISvm> RandomSearchWorkflow::run()
{
    //find number of Tries from run of SSVM the same for support vector number and feature number
    auto outPath = m_config.outputFolderPath.string();
    auto evoInfo = numberOfTries(outPath);

    // put into population
    m_pop = initRandom(evoInfo.m_tries);
    
    // evaluate
    m_svmTraining.launch(m_pop, m_trainingSet);

    m_popTestSet = m_pop;   //copy in here!!!
    m_validation.launch(m_pop, m_validationSet);
    m_validation.launch(m_popTestSet, m_testSet);

    // save the results to logger
    log();
    m_resultLogger.logToFile(m_resultFilePath);
    
    //finish
    return m_pop.getBestOne().getClassifier();
}


void RandomSearchWorkflow::log()
{
    //TODO add header to logs and modify python analysis to use this header
    auto bestOneConfustionMatrix = m_pop.getBestOne().getConfusionMatrix().value();
    //auto featureNumber = m_validationSet.getSamples()[0].size();

    m_resultLogger.createLogEntry(m_pop,
                                  m_popTestSet,
                                  m_timer,
                                  m_algorithmName,
                                  m_generationNumber,
                                  svmComponents::Accuracy(bestOneConfustionMatrix),
                                  m_pop.getBestOne().featureSetSize(),
                                  //m_numberOfClassExamples * m_algorithmConfig.m_labelsCount.size(),
                                  bestOneConfustionMatrix);

}




RandomSearchWithInitialPopulationsWorkflow::RandomSearchWithInitialPopulationsWorkflow(const SvmWokrflowConfiguration& config,
                                                                                       RandomSearchWorkflowInitPopsConfig algorithmConfig,
                                                                                       IDatasetLoader& workflow)
    : m_resultFilePath(std::filesystem::path(config.outputFolderPath.string() + config.txtLogFilename))
    , m_algorithmConfig(algorithmConfig)
    , m_svmTraining(*m_algorithmConfig.m_svmTraining)
    , m_workflow(workflow)
    , m_config(config)
    , m_generationNumber(0)
    , m_trainingSet(workflow.getTraningSet())
    , m_validationSet(workflow.getValidationSet())
    , m_testSet(workflow.getTestSet())
    , m_validation(*algorithmConfig.m_svmConfig.m_estimationMethod, false)
    , m_validationTest(*algorithmConfig.m_svmConfig.m_estimationMethod, true)
    , m_trainingSetOptimization(std::move(algorithmConfig.m_trainingSetOptimization))
    , m_kernelOptimization(std::move(algorithmConfig.m_kernelOptimization))
    , m_featureSetOptimization(std::move(algorithmConfig.m_featureSetOptimization))
{
}

geneticComponents::Population<svmComponents::SvmSimultaneousChromosome> RandomSearchWithInitialPopulationsWorkflow::initRandom(int numberOfTries)
{
    auto popSize = numberOfTries;
    auto features = m_featureSetOptimization->initNoEvaluate(popSize);
    auto kernels = m_kernelOptimization->initNoEvaluate(popSize);
    auto traningSets = m_trainingSetOptimization->initNoEvaluate(popSize);

    std::vector<svmComponents::SvmSimultaneousChromosome> vec;
    vec.reserve(popSize);
    for (auto i = 0u; i < features.size(); ++i)
    {
        svmComponents::SvmSimultaneousChromosome c{ kernels[i], traningSets[i], features[i] };
        vec.emplace_back(c);
    }
    geneticComponents::Population<svmComponents::SvmSimultaneousChromosome> pop{ vec };
    return pop;
}

std::shared_ptr<phd::svm::ISvm> RandomSearchWithInitialPopulationsWorkflow::run()
{
    //find number of Tries from run of SSVM the same for support vector number and feature number
    auto outPath = m_config.outputFolderPath.string();
    auto evoInfo = numberOfTries(outPath);

    // put into population
    m_pop = initRandom(evoInfo.m_tries);

    // evaluate
    m_svmTraining.launch(m_pop, m_trainingSet);

    m_popTestSet = m_pop;   //copy in here!!!
    m_validation.launch(m_pop, m_validationSet);
    m_validation.launch(m_popTestSet, m_testSet);

    // save the results to logger
    log();
    m_resultLogger.logToFile(m_resultFilePath);

    //finish
    return m_pop.getBestOne().getClassifier();
}

void RandomSearchWithInitialPopulationsWorkflow::log()
{
    //TODO add header to logs and modify python analysis to use this header
    auto bestOneConfustionMatrix = m_pop.getBestOne().getConfusionMatrix().value();
    //auto featureNumber = m_validationSet.getSamples()[0].size();

    m_resultLogger.createLogEntry(m_pop,
                                  m_popTestSet,
                                  m_timer,
                                  m_algorithmName,
                                  m_generationNumber,
                                  Accuracy(bestOneConfustionMatrix),
                                  m_pop.getBestOne().featureSetSize(),
                                  //m_numberOfClassExamples * m_algorithmConfig.m_labelsCount.size(),
                                  bestOneConfustionMatrix);

}

RandomSearchWithHelpFromEvolutionWorkflow::RandomSearchWithHelpFromEvolutionWorkflow(const SvmWokrflowConfiguration& config,
                                                                                     RandomSearchWorkflowEvoHelpConfig algorithmConfig,
                                                                                     IDatasetLoader& workflow)
    : m_resultFilePath(std::filesystem::path(config.outputFolderPath.string() + config.txtLogFilename))
    , m_algorithmConfig(algorithmConfig)
    , m_svmTraining(*m_algorithmConfig.m_svmTraining)
    , m_workflow(workflow)
    , m_config(config)
    , m_generationNumber(0)
    , m_trainingSet(workflow.getTraningSet())
    , m_validationSet(workflow.getValidationSet())
    , m_testSet(workflow.getTestSet())
    , m_validation(*algorithmConfig.m_svmConfig.m_estimationMethod, false)
    , m_validationTest(*algorithmConfig.m_svmConfig.m_estimationMethod, true)
    , m_trainingSetOptimization(std::move(algorithmConfig.m_trainingSetOptimization))
    , m_kernelOptimization(std::move(algorithmConfig.m_kernelOptimization))
    , m_featureSetOptimization(std::move(algorithmConfig.m_featureSetOptimization))
{
}



std::shared_ptr<phd::svm::ISvm> RandomSearchWithHelpFromEvolutionWorkflow::run()
{
    //find number of Tries from run of SSVM the same for support vector number and feature number
    auto outPath = m_config.outputFolderPath.string();
    auto evoInfo = numberOfTries(outPath);

    // put into population
    m_pop = initRandom(evoInfo.m_tries);
    m_popTestSet = m_pop;

    // evaluate
    //auto batch = 20;
    //for(int i = 0; i < m_pop.size() / batch; ++i)
    //{
    //    auto& popVec = m_pop.get();
    //    auto first = popVec.begin() + i * batch;
    //    auto last = popVec.begin() + (i + 1) * batch;
    //    std::vector<svmComponents::SvmSimultaneousChromosome> newVec(first, last);
    //    geneticComponents::Population<svmComponents::SvmSimultaneousChromosome> pop{ newVec };
    //    m_svmTraining.launch(pop, m_trainingSet);
    //    auto popTestSet = pop;   //copy in here!!!
    //    m_validation.launch(pop, m_validationSet);
    //    m_validation.launch(popTestSet, m_testSet);

    //    int j = 0;
    //    for(const auto& individual : pop)
    //    {
    //        m_pop[j + batch * i].updateFitness(individual.getFitness());
    //        m_popTestSet[j + batch * i].updateFitness(popTestSet[j].getFitness());
    //        j++;
    //    }
    //}
    //if(evoInfo.m_tries - (m_pop.size() / batch) * batch > 0)
    //{
    //    auto numberOfBatches = static_cast<int>(m_pop.size() / batch);
    //    auto& popVec = m_pop.get();
    //    auto first = popVec.begin() + numberOfBatches * batch;
    //    auto last = popVec.end();
    //    std::vector<svmComponents::SvmSimultaneousChromosome> newVec(first, last);
    //    geneticComponents::Population<svmComponents::SvmSimultaneousChromosome> pop{ newVec };
    //    m_svmTraining.launch(pop, m_trainingSet);
    //    int j = 0;
    //    for (const auto& individual : pop)
    //    {
    //        m_pop[j++ + batch * numberOfBatches].updateClassifier(individual.getClassifier());
    //    }

    //}

    m_svmTraining.launch(m_pop, m_trainingSet);

    m_popTestSet = m_pop;   //copy in here!!!
    m_validation.launch(m_pop, m_validationSet);
    m_validation.launch(m_popTestSet, m_testSet);

    // save the results to logger
    log();
    m_resultLogger.logToFile(m_resultFilePath);

    //finish
    return m_pop.getBestOne().getClassifier();
}

geneticComponents::Population<svmComponents::SvmSimultaneousChromosome> RandomSearchWithHelpFromEvolutionWorkflow::initRandom(int tries)
{
    const auto maxTraningSetSize = m_trainingSet.size();
    const auto maxFeatureNumber = m_trainingSet.getSample(0).size();

    // "Min": "0.0001",
    // "Max": "1000.1"
    //get kernel
    std::unique_ptr<my_random::IRandomNumberGenerator> rnd = std::make_unique<my_random::MersenneTwister64Rng>(static_cast<unsigned long long>(std::chrono::system_clock::now().time_since_epoch().count()));
    svmComponents::SvmKernelRandomGeneration generation(std::uniform_real_distribution<double>(0.0001, 1000.1), phd::svm::KernelTypes::Rbf, std::move(rnd), false);

    auto kernelPop = generation.createPopulation(tries);

    auto outPath = m_config.outputFolderPath.string();
    auto evoInfo = numberOfTries(outPath);

    //get training set
    constexpr auto numberOfClasses = 2;
    auto trainingPop = createPopulationTrainingSet(tries, m_trainingSet, numberOfClasses, false, evoInfo.m_supportVectorNumber);

    //get feature set
    auto featureSetPopulation = createPopulationFeatures(tries, m_trainingSet, false, evoInfo.m_featureNumber);


    std::vector<svmComponents::SvmSimultaneousChromosome> vec;
    vec.reserve(tries);
    for (auto i = 0; i < tries; ++i)
    {
        svmComponents::SvmSimultaneousChromosome c{ kernelPop[i], trainingPop[i], featureSetPopulation[i] };
        vec.emplace_back(c);
    }
    geneticComponents::Population<svmComponents::SvmSimultaneousChromosome> pop{ vec };
    m_pop = pop;

    return m_pop;

}

void RandomSearchWithHelpFromEvolutionWorkflow::log()
{
    //TODO add header to logs and modify python analysis to use this header
    auto bestOneConfustionMatrix = m_pop.getBestOne().getConfusionMatrix().value();
    //auto featureNumber = m_validationSet.getSamples()[0].size();

    m_resultLogger.createLogEntry(m_pop,
                                  m_popTestSet,
                                  m_timer,
                                  m_algorithmName,
                                  m_generationNumber,
                                  Accuracy(bestOneConfustionMatrix),
                                  m_pop.getBestOne().featureSetSize(),
                                  //m_numberOfClassExamples * m_algorithmConfig.m_labelsCount.size(),
                                  bestOneConfustionMatrix);
}
} // namespace genetic
