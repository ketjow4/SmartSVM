#include "ConfigGeneration.h"

platform::Subtree testApp::ConfigManager::loadTemplateConfig(const std::filesystem::path& configPath)
{
    platform::Subtree config(configPath);

   

    return config;
}

void testApp::ConfigManager::setRandomNumberGenerators(platform::Subtree& config)
{
  /*  static int seed = 0;
    setSeedForRng(config, seed);
    seed++;*/
	
    config.putValue<bool>("Svm.FeatureSetSelection.RandomNumberGenerator.IsSeedRandom", true);
	config.putValue<bool>("Svm.MemeticFeatureSetSelection.RandomNumberGenerator.IsSeedRandom", true);
    config.putValue<bool>("Svm.MemeticTrainingSetSelection.RandomNumberGenerator.IsSeedRandom", true);
    config.putValue<bool>("Svm.GaSvm.RandomNumberGenerator.IsSeedRandom", true);
    config.putValue<bool>("Svm.GeneticKernelEvolution.RandomNumberGenerator.IsSeedRandom", true);
	config.putValue<bool>("Svm.CustomKernel.RandomNumberGenerator.IsSeedRandom", true);
	config.putValue<bool>("Svm.SequentialGamma.RandomNumberGenerator.IsSeedRandom", true);
	config.putValue<bool>("Svm.MultipleGammaMASVM.RandomNumberGenerator.IsSeedRandom", true);
    config.putValue<bool>("Svm.RbfLinear.RandomNumberGenerator.IsSeedRandom", true);
}

void testApp::ConfigManager::setSeedForRng(platform::Subtree& config, int seed)
{
    config.putValue<int>("Svm.FeatureSetSelection.RandomNumberGenerator.Seed", seed);
    config.putValue<int>("Svm.MemeticFeatureSetSelection.RandomNumberGenerator.Seed", seed);
    config.putValue<int>("Svm.MemeticTrainingSetSelection.RandomNumberGenerator.Seed", seed);
    config.putValue<int>("Svm.GaSvm.RandomNumberGenerator.Seed", seed);
    config.putValue<int>("Svm.GeneticKernelEvolution.RandomNumberGenerator.Seed", seed);
    config.putValue<int>("Svm.CustomKernel.RandomNumberGenerator.Seed", seed);
    config.putValue<int>("Svm.MultipleGammaMASVM.RandomNumberGenerator.Seed", seed);
    config.putValue<int>("Svm.SequentialGamma.RandomNumberGenerator.Seed", seed);
}

void testApp::ConfigManager::setMetric(platform::Subtree& config)
{
	if(config.getValue<std::string>("Name") == "RbfLinear" || config.getValue<std::string>("Name") == "RbfLinearCoevolution")
	{
        config.putValue<std::string>("Svm.Metric", "BalancedAccuracy"); //linear case starts with accuracy
        //config.putValue<std::string>("Svm.Metric", "AUC"); //linear case starts with accuracy
	}
    else if (config.getValue<std::string>("Name") ==  "EnsembleTree")
    {
        config.putValue<std::string>("Svm.Metric", "CertainAccuracy");
    }
    else
    {
        config.putValue<std::string>("Svm.Metric", "AUC");
       // config.putValue<std::string>("Svm.Metric", "MCC");
    }
    //config.putValue<std::string>("Svm.Metric", "Accuracy");
    config.putValue<int>("GridSearch.NumberOfIteratrions", 1);
}


void testApp::ConfigManager::setMetricRegression(platform::Subtree& config)
{
    config.putValue<std::string>("Svm.Metric", "R2");
    config.putValue<int>("GridSearch.NumberOfIteratrions", 3);
}

void testApp::ConfigManager::setNumberOfClasses(platform::Subtree& config, uint64_t numberOfClasses)
{
    config.putValue<unsigned int>("Svm.GaSvm.NumberOfClasses", numberOfClasses);
    config.putValue<unsigned int>("Svm.MemeticTrainingSetSelection.NumberOfClasses", numberOfClasses);
    config.putValue<unsigned int>("Svm.MemeticFeatureSetSelection.NumberOfClasses", numberOfClasses);
    //config.putValue<unsigned int>("Svm.CustomKernel.NumberOfClasses", numberOfClasses);  This algorithm should not be used
    config.putValue<unsigned int>("Svm.SequentialGamma.NumberOfClasses", numberOfClasses);
    config.putValue<unsigned int>("Svm.MultipleGammaMASVM.NumberOfClasses", numberOfClasses);
}

void testApp::ConfigManager::setGridKernelInitialPopulationGeneration(platform::Subtree& config)
{
    config.putValue<std::string>("Svm.GeneticKernelEvolution.Generation.Name", "Grid");
    config.putValue<double>("Svm.GeneticKernelEvolution.Generation.Grid.Min", 0.001);
    config.putValue<double>("Svm.GeneticKernelEvolution.Generation.Grid.Max", 1000.1);
}

void testApp::ConfigManager::addKernelParameters(platform::Subtree& config, double C, double gamma)
{
    config.putValue<double>("Svm.GaSvm.Kernel.Gamma", gamma);
    config.putValue<double>("Svm.GaSvm.Kernel.C", C);
    config.putValue<double>("Svm.MemeticTrainingSetSelection.Kernel.Gamma", gamma);
    config.putValue<double>("Svm.MemeticTrainingSetSelection.Kernel.C", C);
    config.putValue<double>("Svm.FeatureSetSelection.Kernel.Gamma", gamma);
    config.putValue<double>("Svm.FeatureSetSelection.Kernel.C", C);
}

void testApp::ConfigManager::setupDataset(platform::Subtree& config, std::string datafolder, std::string algorithmName, std::string resultFolder)
{
    std::string outputFolder = algorithmName;
	if(resultFolder == datafolder)
	{
		config.putValue<std::string>("Svm.OutputFolderPath", datafolder + "\\" + outputFolder + "\\");
	}
    else
    {
        std::vector<std::string> elements = splitPath(datafolder);
	    config.putValue<std::string>("Svm.OutputFolderPath", resultFolder + "\\" + *(elements.end() - 2) + "\\" + *(elements.end() - 1) + "\\" + outputFolder + "\\");
    }

    std::string extension = ".csv";
    std::string treningName = "train";
    std::string validationName = "validation";
    std::string testName = "test";

    //filesystem::FileSystem fs;
    if (std::filesystem::exists(datafolder + "\\train.csv"))
    {
        ;
    }
    else
    {
        extension = ".groups";
    }
    config.putValue<std::string>("Svm.ValidationData", datafolder + "\\" + validationName + extension);
    config.putValue<std::string>("Svm.TestData", datafolder + "\\" + testName + extension);
    config.putValue<std::string>("Svm.TrainingData", datafolder + "\\" + treningName + extension);
}

void testApp::ConfigManager::setK(platform::Subtree& config, uint32_t K)
{
    config.putValue<unsigned int>("Svm.GaSvm.NumberOfClassExamples", K);
    config.putValue<unsigned int>("Svm.MemeticTrainingSetSelection.NumberOfClassExamples", K);
	config.putValue<unsigned int>("Svm.CustomKernel.NumberOfClassExamples", K);
	config.putValue<unsigned int>("GridSearch.SubsetSize", K);
}

void testApp::ConfigManager::setInitialNumberOfFeatures(platform::Subtree& config, uint32_t features)
{
    config.putValue("Svm.MemeticFeatureSetSelection.NumberOfClassExamples", features);
}

void testApp::ConfigManager::saveConfigToFileFolds(const platform::Subtree& config, std::string datafolder, std::string algorithmName, uint32_t fold)
{
    if(!std::filesystem::exists(datafolder))
    {
        auto areCreated = std::filesystem::create_directories(datafolder);
    }
	
    static int i = 0;
    //platform::ConfigParser<platform::JSON> m_parser;
    //m_parser.dump(datafolder + "\\" + /*std::to_string(i)  +*/ algorithmName  + std::to_string(fold) + ".json", config);
    config.save(datafolder + "\\" + /*std::to_string(i)  +*/ algorithmName  + std::to_string(fold) + ".json");
    i++;
}

void testApp::ConfigManager::setupStopCondition(platform::Subtree& config)
{
    auto epsilon = 1e-3;

    config.putValue<std::string>("Svm.GeneticKernelEvolution.StopCondition.Name", "MeanFitness");
    config.putValue<double>("Svm.GeneticKernelEvolution.StopCondition.MeanFitness.Epsilon", epsilon);

    config.putValue<std::string>("Svm.GaSvm.StopCondition.Name", "MeanFitness");
    config.putValue<double>("Svm.GaSvm.StopCondition.MeanFitness.Epsilon", epsilon);

    config.putValue<std::string>("Svm.MemeticTrainingSetSelection.StopCondition.Name", "MeanFitness");
    config.putValue<double>("Svm.MemeticTrainingSetSelection.StopCondition.MeanFitness.Epsilon", epsilon);

    config.putValue<std::string>("Svm.FeatureSetSelection.StopCondition.Name", "MeanFitness");
    config.putValue<double>("Svm.FeatureSetSelection.StopCondition.MeanFitness.Epsilon", epsilon);

    config.putValue<std::string>("Svm.MemeticFeatureSetSelection.StopCondition.Name", "MeanFitness");
    config.putValue<double>("Svm.MemeticFeatureSetSelection.StopCondition.MeanFitness.Epsilon", epsilon);

    config.putValue<std::string>("Svm.CustomKernel.StopCondition.Name", "MeanFitness");
    config.putValue<double>("Svm.CustomKernel.StopCondition.MeanFitness.Epsilon", epsilon);

	config.putValue<std::string>("Svm.SequentialGamma.StopCondition.Name", "MeanFitness");
	config.putValue<double>("Svm.SequentialGamma.StopCondition.MeanFitness.Epsilon", epsilon);
}
