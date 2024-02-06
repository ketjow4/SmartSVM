
#include "DefaultWorkflowConfigs.h"

namespace genetic
{
platform::Subtree DefaultSvmConfig::getDefault()
{
    platform::Subtree config;

    config.putValue("Svm.KernelType", "RBF");
    config.putValue<bool>("Svm.UseSvmIteration", false);
    config.putValue<int>("Svm.SvmIterationNumber", 0);
    config.putValue<double>("Svm.Epsilon", 0.001);
    config.putValue<std::string>("Svm.OutputFolderPath", ".\\");
    config.putValue<std::string>("Svm.Visualization.Filename", "visualization.png");
    config.putValue<std::string>("Svm.TxtLogFilename", "results.txt");
    config.putValue<bool>("Svm.Visualization.Create", true);
    config.putValue<int>("Svm.Visualization.Width", 500);
    config.putValue<int>("Svm.Visualization.Height", 500);
    config.putValue<std::string>("Svm.Metric", "Accuracy");
    config.putValue<std::string>("Svm.Type", "LibSvm");  //OpenCvSvm  LibSvm
    config.putValue<bool>("Svm.isRegression", false);

    config.putValue<std::string>("Svm.LogVerbosity", "Standard");
    

    // @wdudzik paths to input datasets
    config.putValue<std::string>("Svm.ValidationData", R"(validation.csv)");
    config.putValue<std::string>("Svm.TestData", R"(test.csv)");
    config.putValue<std::string>("Svm.TrainingData", R"(trening.csv)");

    return config;
}

platform::Subtree DefaultSvmConfig::getRegressionSvr()
{
    platform::Subtree config;

    config.putValue("Svm.KernelType", "RBF");
    config.putValue<bool>("Svm.UseSvmIteration", false);
    config.putValue<int>("Svm.SvmIterationNumber", 0);
    config.putValue<double>("Svm.Epsilon", 0.001);
    config.putValue<std::string>("Svm.OutputFolderPath", ".\\");
    config.putValue<std::string>("Svm.Visualization.Filename", "visualization.png");
    config.putValue<std::string>("Svm.TxtLogFilename", "results.txt");
    config.putValue<bool>("Svm.Visualization.Create", false);
    config.putValue<int>("Svm.Visualization.Width", 0);
    config.putValue<int>("Svm.Visualization.Height", 0);
    config.putValue<std::string>("Svm.Metric", "R2");
    config.putValue<std::string>("Svm.Type", "LibSvm");


    // @wdudzik paths to input datasets
    config.putValue<std::string>("Svm.ValidationData", R"(validation.csv)");
    config.putValue<std::string>("Svm.TestData", R"(test.csv)");
    config.putValue<std::string>("Svm.TrainingData", R"(trening.csv)");

    return config;
}

platform::Subtree DefaultKernelEvolutionConfig::getDefault()
{
    auto config = DefaultSvmConfig::getDefault();

    config.putValue("Name", "GeneticKernelEvolution");
    config.putValue<unsigned int>("Svm.GeneticKernelEvolution.PopulationSize", 20);
    config.putValue<bool>("Svm.GeneticKernelEvolution.Svm.isRegression", false);

    // @wdudzik Crossover
    config.putValue<std::string>("Svm.GeneticKernelEvolution.Crossover.Name", "Heuristic");
    config.putValue<double>("Svm.GeneticKernelEvolution.Crossover.Heuristic.AlphaRange.Min", 0.5);
    config.putValue<double>("Svm.GeneticKernelEvolution.Crossover.Heuristic.AlphaRange.Max", 1.5);

    // @wdudzik Mutation
    config.putValue<std::string>("Svm.GeneticKernelEvolution.Mutation.Name", "ParameterMutation");
    config.putValue<double>("Svm.GeneticKernelEvolution.Mutation.ParameterMutation.Probability", 0.1);
    config.putValue<double>("Svm.GeneticKernelEvolution.Mutation.ParameterMutation.MaxMutationChangeInPercent", 0.1);

    // @wdudzik Population generation
    config.putValue<std::string>("Svm.GeneticKernelEvolution.Generation.Name", "Grid");
    config.putValue<double>("Svm.GeneticKernelEvolution.Generation.Grid.Min", 0.0001);
    config.putValue<double>("Svm.GeneticKernelEvolution.Generation.Grid.Max", 1000.1);
    config.putValue<bool>("Svm.GeneticKernelEvolution.Generation.isRegression", false);

   

    // @wdudzik Stop condition
    config.putValue<std::string>("Svm.GeneticKernelEvolution.StopCondition.Name", "MeanFitness");
    config.putValue<double>("Svm.GeneticKernelEvolution.StopCondition.MeanFitness.Epsilon", 1e-6);

    // @wdudzik Selection
    config.putValue<std::string>("Svm.GeneticKernelEvolution.SelectionOperator.Name", "TruncationSelection");
    config.putValue<double>("Svm.GeneticKernelEvolution.SelectionOperator.TruncationSelection.TruncationCoefficient", 0.5);

    // @wdudzik Crossover selection
    config.putValue<std::string>("Svm.GeneticKernelEvolution.CrossoverSelection.Name", "HighLowFit");
    config.putValue<double>("Svm.GeneticKernelEvolution.CrossoverSelection.HighLowFit.HighLowCoefficient", 0.5);

    // @wdudzik Random number generator
    config.putValue<std::string>("Svm.GeneticKernelEvolution.RandomNumberGenerator.Name", "Mt_19937");
    config.putValue<bool>("Svm.GeneticKernelEvolution.RandomNumberGenerator.IsSeedRandom", false);
    config.putValue<int>("Svm.GeneticKernelEvolution.RandomNumberGenerator.Seed", 0);
    return config;
}

platform::Subtree DefaultKernelEvolutionConfig::getRegressionSvr()
{
    auto config = DefaultSvmConfig::getRegressionSvr();
    config.putValue<bool>("Svm.GeneticKernelEvolution.Svm.isRegression", true);


    config.putValue("Name", "GeneticKernelEvolution");
    config.putValue<unsigned int>("Svm.GeneticKernelEvolution.PopulationSize", 20);

    // @wdudzik Crossover
    config.putValue<std::string>("Svm.GeneticKernelEvolution.Crossover.Name", "Heuristic");
    config.putValue<double>("Svm.GeneticKernelEvolution.Crossover.Heuristic.AlphaRange.Min", 0.5);
    config.putValue<double>("Svm.GeneticKernelEvolution.Crossover.Heuristic.AlphaRange.Max", 1.5);

    // @wdudzik Mutation
    config.putValue<std::string>("Svm.GeneticKernelEvolution.Mutation.Name", "ParameterMutation");
    config.putValue<double>("Svm.GeneticKernelEvolution.Mutation.ParameterMutation.Probability", 0.1);
    config.putValue<double>("Svm.GeneticKernelEvolution.Mutation.ParameterMutation.MaxMutationChangeInPercent", 0.1);

    // @wdudzik Population generation
    config.putValue<std::string>("Svm.GeneticKernelEvolution.Generation.Name", "RandomInRange");
    config.putValue<double>("Svm.GeneticKernelEvolution.Generation.RandomInRange.Min", 0.0001);
    config.putValue<double>("Svm.GeneticKernelEvolution.Generation.RandomInRange.Max", 1000.1);
    config.putValue<bool>("Svm.GeneticKernelEvolution.Generation.isRegression", true);

    // @wdudzik Stop condition
    config.putValue<std::string>("Svm.GeneticKernelEvolution.StopCondition.Name", "MeanFitness");
    config.putValue<double>("Svm.GeneticKernelEvolution.StopCondition.MeanFitness.Epsilon", 1e-6);

    // @wdudzik Selection
    config.putValue<std::string>("Svm.GeneticKernelEvolution.SelectionOperator.Name", "TruncationSelection");
    config.putValue<double>("Svm.GeneticKernelEvolution.SelectionOperator.TruncationSelection.TruncationCoefficient", 0.5);

    // @wdudzik Crossover selection
    config.putValue<std::string>("Svm.GeneticKernelEvolution.CrossoverSelection.Name", "HighLowFit");
    config.putValue<double>("Svm.GeneticKernelEvolution.CrossoverSelection.HighLowFit.HighLowCoefficient", 0.5);

    // @wdudzik Random number generator
    config.putValue<std::string>("Svm.GeneticKernelEvolution.RandomNumberGenerator.Name", "Mt_19937");
    config.putValue<bool>("Svm.GeneticKernelEvolution.RandomNumberGenerator.IsSeedRandom", false);
    config.putValue<int>("Svm.GeneticKernelEvolution.RandomNumberGenerator.Seed", 0);
    return config;
}

platform::Subtree DefaultGaSvmConfig::getDefault()
{
    auto config = DefaultSvmConfig::getDefault();

    config.putValue("Name", "GaSvm");
    config.putValue<bool>("Svm.GaSvm.Svm.isRegression", false);

    config.putValue<unsigned int>("Svm.GaSvm.PopulationSize", 20);
    config.putValue<unsigned int>("Svm.GaSvm.NumberOfClasses", 2);
    config.putValue<double>("Svm.GaSvm.Kernel.Gamma", 1.0);
    config.putValue<double>("Svm.GaSvm.Kernel.C", 1.0);
    config.putValue<double>("Svm.GaSvm.Kernel.Coef0", 1.0);
    config.putValue<double>("Svm.GaSvm.Kernel.Degree", 3.0);
    config.putValue<unsigned int>("Svm.GaSvm.NumberOfClassExamples", 8); // @wdudzik K parameter from GASVM article

    // @wdudzik Random number generator
    config.putValue<std::string>("Svm.GaSvm.RandomNumberGenerator.Name", "Mt_19937");
    config.putValue<bool>("Svm.GaSvm.RandomNumberGenerator.IsSeedRandom", false);
    config.putValue<int>("Svm.GaSvm.RandomNumberGenerator.Seed", 0);

    // @wdudzik Population generation
    config.putValue<std::string>("Svm.GaSvm.Generation.Name", "Random");

    // @wdudzik Crossover
    config.putValue<std::string>("Svm.GaSvm.Crossover.Name", "GaSvm");

    // @wdudzik Mutation
    config.putValue<std::string>("Svm.GaSvm.Mutation.Name", "GaSvm");
    config.putValue<double>("Svm.GaSvm.Mutation.GaSvm.ExchangePercent", 0.3);
    config.putValue<double>("Svm.GaSvm.Mutation.GaSvm.MutationProbability", 0.3);

    // @wdudzik Stop condition
    config.putValue<std::string>("Svm.GaSvm.StopCondition.Name", "MeanFitness");
    config.putValue<double>("Svm.GaSvm.StopCondition.MeanFitness.Epsilon", 1e-6);

    // @wdudzik Selection
    config.putValue<std::string>("Svm.GaSvm.SelectionOperator.Name", "TruncationSelection");
    config.putValue<double>("Svm.GaSvm.SelectionOperator.TruncationSelection.TruncationCoefficient", 0.5);

    // @wdudzik Crossover selection
    config.putValue<std::string>("Svm.GaSvm.CrossoverSelection.Name", "HighLowFit");
    config.putValue<double>("Svm.GaSvm.CrossoverSelection.HighLowFit.HighLowCoefficient", 0.5);

    config.putValue<std::string>("Svm.GaSvm.Validation.Name", "Regular");
    config.putValue<bool>("Svm.GaSvm.EnhanceTrainingSet", false);

    return config;
}

platform::Subtree DefaultGaSvmConfig::getRegressionSvr()
{
    auto config = DefaultSvmConfig::getRegressionSvr();

    config.putValue("Name", "GaSvm");
    config.putValue<bool>("Svm.GaSvm.Svm.isRegression", true);

    config.putValue<unsigned int>("Svm.GaSvm.PopulationSize", 20);
    config.putValue<unsigned int>("Svm.GaSvm.NumberOfClasses", 1);
    config.putValue<double>("Svm.GaSvm.Kernel.Gamma", 1.0);  
    config.putValue<double>("Svm.GaSvm.Kernel.C", 1.0);
    config.putValue<double>("Svm.GaSvm.Kernel.Coef0", 1.0);
    config.putValue<double>("Svm.GaSvm.Kernel.Degree", 3.0);
    config.putValue<double>("Svm.GaSvm.Kernel.Epsilon", 0.001);
    
    config.putValue<unsigned int>("Svm.GaSvm.NumberOfClassExamples", 8); // @wdudzik K parameter from GASVM article

                                                                         // @wdudzik Random number generator
    config.putValue<std::string>("Svm.GaSvm.RandomNumberGenerator.Name", "Mt_19937");
    config.putValue<bool>("Svm.GaSvm.RandomNumberGenerator.IsSeedRandom", false);
    config.putValue<int>("Svm.GaSvm.RandomNumberGenerator.Seed", 0);

    // @wdudzik Population generation
    config.putValue<std::string>("Svm.GaSvm.Generation.Name", "GaSvmRegression");

    // @wdudzik Crossover
    config.putValue<std::string>("Svm.GaSvm.Crossover.Name", "GaSvmRegression");

    // @wdudzik Mutation
    config.putValue<std::string>("Svm.GaSvm.Mutation.Name", "GaSvmRegression");
    config.putValue<double>("Svm.GaSvm.Mutation.GaSvm.ExchangePercent", 0.3);
    config.putValue<double>("Svm.GaSvm.Mutation.GaSvm.MutationProbability", 0.3);

    // @wdudzik Stop condition
    config.putValue<std::string>("Svm.GaSvm.StopCondition.Name", "MeanFitness");
    config.putValue<double>("Svm.GaSvm.StopCondition.MeanFitness.Epsilon", 1e-6);

    // @wdudzik Selection
    config.putValue<std::string>("Svm.GaSvm.SelectionOperator.Name", "TruncationSelection");
    config.putValue<double>("Svm.GaSvm.SelectionOperator.TruncationSelection.TruncationCoefficient", 0.5);

    // @wdudzik Crossover selection
    config.putValue<std::string>("Svm.GaSvm.CrossoverSelection.Name", "HighLowFit");
    config.putValue<double>("Svm.GaSvm.CrossoverSelection.HighLowFit.HighLowCoefficient", 0.5);

	config.putValue<std::string>("Svm.GaSvm.Validation.Name", "Regular");
    config.putValue<bool>("Svm.GaSvm.EnhanceTrainingSet", false);
	
    return config;
}

platform::Subtree DefaultMemeticConfig::getDefault()
{
    auto config = DefaultSvmConfig::getDefault();

    config.putValue("Name", "MemeticTrainingSetSelection");
    config.putValue<bool>("Svm.MemeticTrainingSetSelection.Svm.isRegression", false);

    config.putValue<unsigned int>("Svm.MemeticTrainingSetSelection.PopulationSize", 10);
    config.putValue<unsigned int>("Svm.MemeticTrainingSetSelection.NumberOfClasses", 2);
    config.putValue<double>("Svm.MemeticTrainingSetSelection.Kernel.Gamma", 1.0);
    config.putValue<double>("Svm.MemeticTrainingSetSelection.Kernel.C", 1.0);
    config.putValue<double>("Svm.MemeticTrainingSetSelection.Kernel.Coef0", 1.0);
    config.putValue<double>("Svm.MemeticTrainingSetSelection.Kernel.Degree", 3.0);
    config.putValue<unsigned int>("Svm.MemeticTrainingSetSelection.NumberOfClassExamples", 8);
    //config.putValue<unsigned int>("Svm.MemeticTrainingSetSelection.NumberOfClassExamples", 32); //for single test only

    // @wdudzik Memetic specific values
    auto superIndividualAlpha = 0.2;
    config.putValue<double>("Svm.MemeticTrainingSetSelection.Memetic.SuperIndividualsAlpha", superIndividualAlpha);
    config.putValue<double>("Svm.MemeticTrainingSetSelection.Memetic.PercentOfSupportVectorsThreshold", 0.3);
    config.putValue<unsigned int>("Svm.MemeticTrainingSetSelection.Memetic.IterationsBeforeModeChange", 3);
    config.putValue<double>("Svm.MemeticTrainingSetSelection.Memetic.EducationProbability", 0.3);
    config.putValue<double>("Svm.MemeticTrainingSetSelection.Memetic.ThresholdForMaxNumberOfClassExamples", 1.0);
    config.putValue<double>("Svm.MemeticTrainingSetSelection.Memetic.MaxK", 0);

    // @wdudzik Random number generator
    config.putValue<std::string>("Svm.MemeticTrainingSetSelection.RandomNumberGenerator.Name", "Mt_19937");
    config.putValue<bool>("Svm.MemeticTrainingSetSelection.RandomNumberGenerator.IsSeedRandom", false);
    config.putValue<int>("Svm.MemeticTrainingSetSelection.RandomNumberGenerator.Seed", 0);

    // @wdudzik Stop condition
    config.putValue<std::string>("Svm.MemeticTrainingSetSelection.StopCondition.Name", "MeanFitness");
    config.putValue<double>("Svm.MemeticTrainingSetSelection.StopCondition.MeanFitness.Epsilon", 1e-6);

    // @wdudzik Selection
    auto truncationCoefficient = 1 / (2 + superIndividualAlpha);
    config.putValue<std::string>("Svm.MemeticTrainingSetSelection.SelectionOperator.Name", "TruncationSelection");
    config.putValue<double>("Svm.MemeticTrainingSetSelection.SelectionOperator.TruncationSelection.TruncationCoefficient", truncationCoefficient);

    // @wdudzik Population generation
    config.putValue<std::string>("Svm.MemeticTrainingSetSelection.Generation.Name", "Random");

    // @wdudzik Crossover selection
    config.putValue<std::string>("Svm.MemeticTrainingSetSelection.CrossoverSelection.Name", "LocalGlobalSelection");
    config.putValue<double>("Svm.MemeticTrainingSetSelection.CrossoverSelection.LocalGlobalSelection.HighLowCoefficient", 0.5);
    config.putValue<bool>("Svm.MemeticTrainingSetSelection.CrossoverSelection.LocalGlobalSelection.IsLocalMode", false);

    // @wdudzik Crossover
    config.putValue<std::string>("Svm.MemeticTrainingSetSelection.Crossover.Name", "Memetic");

    // @wdudzik Mutation
    config.putValue<std::string>("Svm.MemeticTrainingSetSelection.Mutation.Name", "GaSvm");
    config.putValue<double>("Svm.MemeticTrainingSetSelection.Mutation.GaSvm.ExchangePercent", 0.3);
    config.putValue<double>("Svm.MemeticTrainingSetSelection.Mutation.GaSvm.MutationProbability", 0.3);


	config.putValue<std::string>("Svm.MemeticTrainingSetSelection.Validation.Name", "Regular");
	//config.putValue<std::string>("Svm.MemeticTrainingSetSelection.Validation.Name", "Subset");

	config.putValue<bool>("Svm.MemeticTrainingSetSelection.EnhanceTrainingSet", false);

    return config;
}

platform::Subtree DefaultFeatureSelectionConfig::getDefault()
{
    auto config = DefaultSvmConfig::getDefault();

    config.putValue("Name", "FeatureSetSelection");
    config.putValue<unsigned int>("Svm.FeatureSetSelection.PopulationSize", 20);
    config.putValue<double>("Svm.FeatureSetSelection.Kernel.Gamma", 1.0);
    config.putValue<double>("Svm.FeatureSetSelection.Kernel.C", 1.0);
    config.putValue<double>("Svm.FeatureSetSelection.Kernel.Coef0", 1.0);
    config.putValue<double>("Svm.FeatureSetSelection.Kernel.Degree", 3.0);
    config.putValue<bool>("Svm.FeatureSetSelection.Svm.isRegression", false);

    // @wdudzik Crossover
    config.putValue<std::string>("Svm.FeatureSetSelection.Crossover.Name", "FeaturesSelectionOnePoint");

    // @wdudzik Mutation
    config.putValue<std::string>("Svm.FeatureSetSelection.Mutation.Name", "FeaturesSelectionBitFlip");
    config.putValue<double>("Svm.FeatureSetSelection.Mutation.BitFlipProbability", 0.01);

    // @wdudzik Population generation
    config.putValue<std::string>("Svm.FeatureSetSelection.Generation.Name", "FeaturesSelectionRandom");
    config.putValue<double>("Svm.FeatureSetSelection.Generation.PercentageOfFill", 0.1);

    // @wdudzik Random number generator
    config.putValue<std::string>("Svm.FeatureSetSelection.RandomNumberGenerator.Name", "Mt_19937");
    config.putValue<bool>("Svm.FeatureSetSelection.RandomNumberGenerator.IsSeedRandom", false);
    config.putValue<int>("Svm.FeatureSetSelection.RandomNumberGenerator.Seed", 0);

    // @wdudzik Stop condition
    config.putValue<std::string>("Svm.FeatureSetSelection.StopCondition.Name", "MeanFitness");
    config.putValue<double>("Svm.FeatureSetSelection.StopCondition.MeanFitness.Epsilon", 1e-6);

    // @wdudzik Selection
    config.putValue<std::string>("Svm.FeatureSetSelection.SelectionOperator.Name", "TruncationSelection");
    config.putValue<double>("Svm.FeatureSetSelection.SelectionOperator.TruncationSelection.TruncationCoefficient", 0.34);

    // @wdudzik Crossover selection
    config.putValue<std::string>("Svm.FeatureSetSelection.CrossoverSelection.Name", "HighLowFit");
    config.putValue<double>("Svm.FeatureSetSelection.CrossoverSelection.HighLowFit.HighLowCoefficient", 0.5);

    return config;
}

platform::Subtree DefaultFeaturesMemeticConfig::getDefault()
{
    auto config = DefaultSvmConfig::getDefault();

    config.putValue("Name", "MemeticFeatureSetSelection");
    config.putValue<bool>("Svm.MemeticFeatureSetSelection.Svm.isRegression", false);

    config.putValue<unsigned int>("Svm.MemeticFeatureSetSelection.PopulationSize", 10);
    config.putValue<unsigned int>("Svm.MemeticFeatureSetSelection.NumberOfClasses", 2);
    config.putValue<double>("Svm.MemeticFeatureSetSelection.Kernel.Gamma", 1.0);
    config.putValue<double>("Svm.MemeticFeatureSetSelection.Kernel.C", 1.0);
    config.putValue<double>("Svm.MemeticFeatureSetSelection.Kernel.Coef0", 1.0);
    config.putValue<double>("Svm.MemeticFeatureSetSelection.Kernel.Degree", 3.0);
    config.putValue<unsigned int>("Svm.MemeticFeatureSetSelection.NumberOfClassExamples", 4);

    // @wdudzik Memetic specific values
    auto superIndividualAlpha = 0.2;
    config.putValue<double>("Svm.MemeticFeatureSetSelection.Memetic.SuperIndividualsAlpha", superIndividualAlpha);
    config.putValue<double>("Svm.MemeticFeatureSetSelection.Memetic.PercentOfSupportVectorsThreshold", 0.3);
    config.putValue<unsigned int>("Svm.MemeticFeatureSetSelection.Memetic.IterationsBeforeModeChange", 3);
    config.putValue<double>("Svm.MemeticFeatureSetSelection.Memetic.EducationProbability", 0.3);
    config.putValue<double>("Svm.MemeticFeatureSetSelection.Memetic.ThresholdForMaxNumberOfClassExamples", 1.0);

    // @wdudzik Random number generator
    config.putValue<std::string>("Svm.MemeticFeatureSetSelection.RandomNumberGenerator.Name", "Mt_19937");
    config.putValue<bool>("Svm.MemeticFeatureSetSelection.RandomNumberGenerator.IsSeedRandom", false);
    config.putValue<int>("Svm.MemeticFeatureSetSelection.RandomNumberGenerator.Seed", 0);

    // @wdudzik Stop condition
    config.putValue<std::string>("Svm.MemeticFeatureSetSelection.StopCondition.Name", "MeanFitness");
    config.putValue<double>("Svm.MemeticFeatureSetSelection.StopCondition.MeanFitness.Epsilon", 1e-6);

    // @wdudzik Selection
    auto truncationCoefficient = 1 / (2 + superIndividualAlpha);
    config.putValue<std::string>("Svm.MemeticFeatureSetSelection.SelectionOperator.Name", "TruncationSelection");
    config.putValue<double>("Svm.MemeticFeatureSetSelection.SelectionOperator.TruncationSelection.TruncationCoefficient", truncationCoefficient);

    // @wdudzik Population generation
    config.putValue<std::string>("Svm.MemeticFeatureSetSelection.Generation.Name", "MutualInfo");

    // @wdudzik Crossover selection
    config.putValue<std::string>("Svm.MemeticFeatureSetSelection.CrossoverSelection.Name", "LocalGlobalSelection");
    config.putValue<double>("Svm.MemeticFeatureSetSelection.CrossoverSelection.LocalGlobalSelection.HighLowCoefficient", 0.5);
    config.putValue<bool>("Svm.MemeticFeatureSetSelection.CrossoverSelection.LocalGlobalSelection.IsLocalMode", false);

    // @wdudzik Crossover
    config.putValue<std::string>("Svm.MemeticFeatureSetSelection.Crossover.Name", "Memetic");

    // @wdudzik Mutation
    config.putValue<std::string>("Svm.MemeticFeatureSetSelection.Mutation.Name", "GaSvm");
    config.putValue<double>("Svm.MemeticFeatureSetSelection.Mutation.GaSvm.ExchangePercent", 0.2);
    config.putValue<double>("Svm.MemeticFeatureSetSelection.Mutation.GaSvm.MutationProbability", 0.5);

    return config;
}

platform::Subtree DefaultAlgaConfig::getDefault()
{
    auto config = DefaultSvmConfig::getDefault();

    config.putValue("Name", "Alga");
    config.putValue<bool>("Svm.Alga.Svm.isRegression", false);

    // @wdudzik Insert default configs for algorithms used in Alga
    config.putNode("Svm.GaSvm", DefaultGaSvmConfig::getDefault().getNode("Svm.GaSvm"));
    config.putNode("Svm.GeneticKernelEvolution", DefaultKernelEvolutionConfig::getDefault().getNode("Svm.GeneticKernelEvolution"));

    // @wdudzik Choose which algorithm will run in Alga
    config.putValue("Svm.Alga.KernelOptimization.Name", "GeneticKernelEvolution");
    config.putValue("Svm.Alga.TrainingSetOptimization.Name", "GaSvm");

    return config;
}

platform::Subtree DefaultAlgaConfig::getALMA()
{
    auto config = DefaultSvmConfig::getDefault();

    config.putValue("Name", "Alma");
    config.putValue<bool>("Svm.Alga.Svm.isRegression", false);

    // @wdudzik Insert default configs for algorithms used in Alga
    config.putNode("Svm.MemeticTrainingSetSelection", DefaultMemeticConfig::getDefault().getNode("Svm.MemeticTrainingSetSelection"));
    config.putNode("Svm.GeneticKernelEvolution", DefaultKernelEvolutionConfig::getDefault().getNode("Svm.GeneticKernelEvolution"));

    // @wdudzik Choose which algorithm will run in Alga
    config.putValue("Svm.Alga.KernelOptimization.Name", "GeneticKernelEvolution");
    config.putValue("Svm.Alga.TrainingSetOptimization.Name", "MemeticTrainingSetSelection");

    return config;
}

platform::Subtree DefaultAlgaConfig::getALGA_regression()
{
    auto config = DefaultSvmConfig::getRegressionSvr();

    config.putValue("Name", "Alga");
    config.putValue<bool>("Svm.Alga.Svm.isRegression", true);

    // @wdudzik Insert default configs for algorithms used in Alga
    config.putNode("Svm.GaSvm", DefaultGaSvmConfig::getRegressionSvr().getNode("Svm.GaSvm"));
    config.putNode("Svm.GeneticKernelEvolution", DefaultKernelEvolutionConfig::getRegressionSvr().getNode("Svm.GeneticKernelEvolution"));

    // @wdudzik Choose which algorithm will run in Alga
    config.putValue("Svm.Alga.KernelOptimization.Name", "GeneticKernelEvolution");
    config.putValue("Svm.Alga.TrainingSetOptimization.Name", "GaSvm");

    return config;
}

platform::Subtree DefaultKTFConfig::getDefault()
{
    auto config = DefaultSvmConfig::getDefault();

    config.putValue("Name", "KTF");
    config.putValue<bool>("Svm.KTF.Svm.isRegression", false);

    // @wdudzik Insert default configs for algorithms used in KTF
    config.putNode("Svm.MemeticTrainingSetSelection", DefaultMemeticConfig::getDefault().getNode("Svm.MemeticTrainingSetSelection"));
    config.putNode("Svm.GeneticKernelEvolution", DefaultKernelEvolutionConfig::getDefault().getNode("Svm.GeneticKernelEvolution"));
    config.putNode("Svm.MemeticFeatureSetSelection", DefaultFeaturesMemeticConfig::getDefault().getNode("Svm.MemeticFeatureSetSelection"));

    // @wdudzik Choose which algorithm will run in KTF
    config.putValue("Svm.KTF.KernelOptimization.Name", "GeneticKernelEvolution");
    config.putValue("Svm.KTF.TrainingSetOptimization.Name", "MemeticTrainingSetSelection");
    config.putValue("Svm.KTF.FeatureSetOptimization.Name", "MemeticFeatureSetSelection");

    return config;
}

void buildTFAlgorihtmsConfig(platform::Subtree& config, const std::string& algorithmName)
{
    config = DefaultSvmConfig::getDefault();

    // @wdudzik Insert default configs for algorithms used in algorithmName
    config.putNode("Svm.MemeticTrainingSetSelection", DefaultMemeticConfig::getDefault().getNode("Svm.MemeticTrainingSetSelection"));
    config.putNode("Svm.MemeticFeatureSetSelection", DefaultFeaturesMemeticConfig::getDefault().getNode("Svm.MemeticFeatureSetSelection"));

    // @wdudzik Choose which algorithm will run in algorithmName
    config.putValue("Svm." + algorithmName + ".TrainingSetOptimization.Name", "MemeticTrainingSetSelection");
    config.putValue("Svm." + algorithmName + ".FeatureSetOptimization.Name", "MemeticFeatureSetSelection");
}

platform::Subtree DefaultTFConfig::getDefault()
{
    platform::Subtree config;

    buildTFAlgorihtmsConfig(config, "TF");
    config.putValue("Name", "TF");
    return config;
}

platform::Subtree DefaultFTConfig::getDefault()
{
    platform::Subtree config;

    buildTFAlgorihtmsConfig(config, "FT");
    config.putValue("Name", "FT");
    return config;
}

platform::Subtree DefaultGridSearchConfig::getDefault()
{
    auto config = DefaultSvmConfig::getDefault();

    config.putValue("Name", "GridSearch");

    config.putValue<int>("GridSearch.NumberOfIteratrions", 1);
    config.putValue<int>("GridSearch.NumberOfFolds", 2);

    // @wdudzik grids define here are for RBF kernel which is default one
    config.putValue<double>("GridSearch.cGrid.Min", 0.001); //0.0000000000000001
    config.putValue<double>("GridSearch.cGrid.Max", 1000.1);  //10000000000000050.1
    config.putValue<double>("GridSearch.cGrid.LogStep", 10);

    config.putValue<double>("GridSearch.gammaGrid.Min", 0.001);
    config.putValue<double>("GridSearch.gammaGrid.Max", 1000.1);
    config.putValue<double>("GridSearch.gammaGrid.LogStep", 10);

	config.putValue<double>("GridSearch.gammaGrid.Min", 0.001);
	config.putValue<double>("GridSearch.gammaGrid.Max", 1000.1);
	config.putValue<double>("GridSearch.gammaGrid.LogStep", 10);

	config.putValue<double>("GridSearch.degreeGrid.Min", 2);
	config.putValue<double>("GridSearch.degreeGrid.Max", 7);
	config.putValue<double>("GridSearch.degreeGrid.LogStep", 10);

	config.putValue<int>("GridSearch.SubsetSize", 0); //0 means full training set, otherwise random subset of T will be selected
	config.putValue<int>("GridSearch.SubsetRepeats", 1); //number of times subsets will be tested (does not apply when using full T)

    return config;
}

const std::vector<std::string>& DefaultGridSearchConfig::getAllGridsNames()
{
    static const std::vector<std::string> paramsNames = {"cGrid", "gammaGrid","pGrid","nuGrid","coefGrid","degreeGrid"};
    return paramsNames;
}

platform::Subtree DefaultSSVMConfig::getDefault()
{
    auto config = DefaultSvmConfig::getDefault();

    config.putValue("Name", "SSVM");
    config.putValue<bool>("Svm.SSVM.Svm.isRegression", false);

    // @wdudzik Insert default configs for algorithms used in KTF
    config.putNode("Svm.MemeticTrainingSetSelection", DefaultMemeticConfig::getDefault().getNode("Svm.MemeticTrainingSetSelection"));
    config.putNode("Svm.GeneticKernelEvolution", DefaultKernelEvolutionConfig::getDefault().getNode("Svm.GeneticKernelEvolution"));
    config.putNode("Svm.MemeticFeatureSetSelection", DefaultFeaturesMemeticConfig::getDefault().getNode("Svm.MemeticFeatureSetSelection"));

    // @wdudzik Choose which algorithm will run in KTF
    config.putValue("Svm.SSVM.KernelOptimization.Name", "GeneticKernelEvolution");
    config.putValue("Svm.SSVM.TrainingSetOptimization.Name", "MemeticTrainingSetSelection");
    config.putValue("Svm.SSVM.FeatureSetOptimization.Name", "MemeticFeatureSetSelection");

	

	//this depends on the size of Memetic Feature set and memetic training set
    auto superIndividualAlpha = 0.2;
	//auto truncationCoefficient = 2 / (2*2 + superIndividualAlpha);
    auto truncationCoefficient = 1 / (2 + superIndividualAlpha);
	config.putValue("Svm.SSVM.SelectionOperator.Name", "TruncationSelection");
	config.putValue<double>("Svm.SSVM.SelectionOperator.TruncationSelection.TruncationCoefficient", truncationCoefficient);
	config.putValue("Svm.SSVM.PopulationSize", 20);

    return config;
}

platform::Subtree DefaultRandomSearchConfig::getDefault()
{
    auto config = DefaultSvmConfig::getDefault();

    config.putValue("Name", "RandomSearch");

    // @wdudzik Insert default configs for algorithms used in KTF
    config.putNode("Svm.MemeticTrainingSetSelection", DefaultMemeticConfig::getDefault().getNode("Svm.MemeticTrainingSetSelection"));
    config.putNode("Svm.GeneticKernelEvolution", DefaultKernelEvolutionConfig::getDefault().getNode("Svm.GeneticKernelEvolution"));
    config.putNode("Svm.MemeticFeatureSetSelection", DefaultFeaturesMemeticConfig::getDefault().getNode("Svm.MemeticFeatureSetSelection"));

    // @wdudzik Choose which algorithm will run in KTF
    config.putValue("Svm.RandomSearch.KernelOptimization.Name", "GeneticKernelEvolution");
    config.putValue("Svm.RandomSearch.TrainingSetOptimization.Name", "MemeticTrainingSetSelection");
    config.putValue("Svm.RandomSearch.FeatureSetOptimization.Name", "MemeticFeatureSetSelection");

    config.putValue("Svm.RandomSearch.PopulationSize", 20);

    return config;
}

platform::Subtree DefaultRandomSearchInitPopConfig::getDefault()
{
    auto config = DefaultSvmConfig::getDefault();

    config.putValue("Name", "RandomSearchInitPop");

    // @wdudzik Insert default configs for algorithms used in KTF
    config.putNode("Svm.MemeticTrainingSetSelection", DefaultMemeticConfig::getDefault().getNode("Svm.MemeticTrainingSetSelection"));
    config.putNode("Svm.GeneticKernelEvolution", DefaultKernelEvolutionConfig::getDefault().getNode("Svm.GeneticKernelEvolution"));
    config.putNode("Svm.MemeticFeatureSetSelection", DefaultFeaturesMemeticConfig::getDefault().getNode("Svm.MemeticFeatureSetSelection"));

    // @wdudzik Choose which algorithm will run in KTF
    config.putValue("Svm.RandomSearchInitPop.KernelOptimization.Name", "GeneticKernelEvolution");
    config.putValue("Svm.RandomSearchInitPop.TrainingSetOptimization.Name", "MemeticTrainingSetSelection");
    config.putValue("Svm.RandomSearchInitPop.FeatureSetOptimization.Name", "MemeticFeatureSetSelection");

    config.putValue("Svm.RandomSearchInitPop.PopulationSize", 20);

    return config;
}

platform::Subtree DefaultRandomSearchEvoHelpConfig::getDefault()
{
    auto config = DefaultSvmConfig::getDefault();

    config.putValue("Name", "RandomSearchEvoHelp");

    // @wdudzik Insert default configs for algorithms used in KTF
    config.putNode("Svm.MemeticTrainingSetSelection", DefaultMemeticConfig::getDefault().getNode("Svm.MemeticTrainingSetSelection"));
    config.putNode("Svm.GeneticKernelEvolution", DefaultKernelEvolutionConfig::getDefault().getNode("Svm.GeneticKernelEvolution"));
    config.putNode("Svm.MemeticFeatureSetSelection", DefaultFeaturesMemeticConfig::getDefault().getNode("Svm.MemeticFeatureSetSelection"));

    // @wdudzik Choose which algorithm will run in KTF
    config.putValue("Svm.RandomSearchEvoHelp.KernelOptimization.Name", "GeneticKernelEvolution");
    config.putValue("Svm.RandomSearchEvoHelp.TrainingSetOptimization.Name", "MemeticTrainingSetSelection");
    config.putValue("Svm.RandomSearchEvoHelp.FeatureSetOptimization.Name", "MemeticFeatureSetSelection");

    config.putValue("Svm.RandomSearchEvoHelp.PopulationSize", 20);

    return config;
}

platform::Subtree CustomKernelConfig::getDefault()
{
    auto config = DefaultSvmConfig::getDefault();

    config.putValue("Name", "CustomKernel");

    config.putValue<bool>("Svm.CustomKernel.Svm.isRegression", false);
    config.putValue<bool>("Svm.CustomKernel.TrainAlpha", true);
    config.putValue<std::string>("Svm.CustomKernel.KernelType", "RBF_CUSTOM");

    config.putValue<unsigned int>("Svm.CustomKernel.PopulationSize", 20);
    config.putValue<unsigned int>("Svm.CustomKernel.NumberOfClasses", 2);
    //config.putValue<double>("Svm.CustomKernel.Kernel.Gamma", 1.0);
    //config.putValue<double>("Svm.CustomKernel.Kernel.C", 1.0);
    config.putValue<unsigned int>("Svm.CustomKernel.NumberOfClassExamples", 8);

    // @wdudzik Memetic specific values
   /* auto superIndividualAlpha = 0.2;
    config.putValue<double>("Svm.MemeticFeatureSetSelection.Memetic.SuperIndividualsAlpha", superIndividualAlpha);
    config.putValue<double>("Svm.MemeticFeatureSetSelection.Memetic.PercentOfSupportVectorsThreshold", 0.3);
    config.putValue<unsigned int>("Svm.MemeticFeatureSetSelection.Memetic.IterationsBeforeModeChange", 3);
    config.putValue<double>("Svm.MemeticFeatureSetSelection.Memetic.EducationProbability", 0.3);
    config.putValue<double>("Svm.MemeticFeatureSetSelection.Memetic.ThresholdForMaxNumberOfClassExamples", 1.0);*/

    // @wdudzik Random number generator
    config.putValue<std::string>("Svm.CustomKernel.RandomNumberGenerator.Name", "Mt_19937");
    config.putValue<bool>("Svm.CustomKernel.RandomNumberGenerator.IsSeedRandom", false);
    config.putValue<int>("Svm.CustomKernel.RandomNumberGenerator.Seed", 0);

    // @wdudzik Stop condition
    config.putValue<std::string>("Svm.CustomKernel.StopCondition.Name", "MeanFitness");
    config.putValue<double>("Svm.CustomKernel.StopCondition.MeanFitness.Epsilon", 1e-6);

    // @wdudzik Selection
    //auto truncationCoefficient = 1 / (2 + superIndividualAlpha);
    auto truncationCoefficient = 1.0 / 2.0;
    config.putValue<std::string>("Svm.CustomKernel.SelectionOperator.Name", "TruncationSelection");
    config.putValue<double>("Svm.CustomKernel.SelectionOperator.TruncationSelection.TruncationCoefficient", truncationCoefficient);

    // @wdudzik Population generation THIS IS HARDCODED
    //config.putValue<std::string>("Svm.CustomKernel.Generation.Name", "MutualInfo");

    //// @wdudzik Crossover selection
    config.putValue<std::string>("Svm.CustomKernel.CrossoverSelection.Name", "LocalGlobalSelection");
    config.putValue<double>("Svm.CustomKernel.CrossoverSelection.LocalGlobalSelection.HighLowCoefficient", 0.5);
    config.putValue<bool>("Svm.CustomKernel.CrossoverSelection.LocalGlobalSelection.IsLocalMode", false);

    //// @wdudzik Crossover
    //config.putValue<std::string>("Svm.CustomKernel.Crossover.Name", "Memetic");

    // @wdudzik Mutation
    config.putValue<std::string>("Svm.CustomKernel.Mutation.Name", "GaSvm");
    config.putValue<double>("Svm.CustomKernel.Mutation.GaSvm.ExchangePercent", 0.3);
    config.putValue<double>("Svm.CustomKernel.Mutation.GaSvm.MutationProbability", 0.3);

    return config;
}

platform::Subtree DefaultSequentialGammaConfig::getDefault()
{
	auto config = DefaultSvmConfig::getDefault();

	config.putValue("Name", "SequentialGamma");

	config.putValue<bool>("Svm.SequentialGamma.Svm.isRegression", false);


    config.putValue<bool>("Svm.SequentialGamma.UseSmallerGamma", true);
    config.putValue<double>("Svm.SequentialGamma.GammaLogStep", 10);
	
    config.putValue<bool>("Svm.SequentialGamma.TrainAlpha", true);
    config.putValue<std::string>("Svm.SequentialGamma.KernelType", "RBF_CUSTOM");
    config.putValue<bool>("Svm.SequentialGamma.ShrinkOnBestOnly", false); //wether shrinking of T should be done based on population or best individual
	

    config.putValue<std::string>("Svm.SequentialGamma.HelperAlgorithmName", "GS");
	
	config.putValue<unsigned int>("Svm.SequentialGamma.PopulationSize", 10);
	config.putValue<unsigned int>("Svm.SequentialGamma.NumberOfClasses", 2);
	config.putValue<unsigned int>("Svm.SequentialGamma.NumberOfClassExamples", 8);

	config.putValue<double>("Svm.SequentialGamma.Kernel.Gamma", 1.0);
	config.putValue<double>("Svm.SequentialGamma.Kernel.C", 1.0);
    config.putValue<double>("Svm.SequentialGamma.Kernel.Coef0", 1.0);
    config.putValue<double>("Svm.SequentialGamma.Kernel.Degree", 3.0);


    config.putValue<std::string>("Svm.SequentialGamma.Validation.Name", "Regular");


	// @wdudzik Memetic specific values
	auto superIndividualAlpha = 0.2;
	config.putValue<double>("Svm.SequentialGamma.SuperIndividualsAlpha", superIndividualAlpha);
	config.putValue<double>("Svm.SequentialGamma.PercentOfSupportVectorsThreshold", 0.3);
	config.putValue<unsigned int>("Svm.SequentialGamma.IterationsBeforeModeChange", 3);
	config.putValue<double>("Svm.SequentialGamma.EducationProbability", 0.3);
	config.putValue<double>("Svm.SequentialGamma.ThresholdForMaxNumberOfClassExamples", 1.0);

	// @wdudzik Random number generator
	config.putValue<std::string>("Svm.SequentialGamma.RandomNumberGenerator.Name", "Mt_19937");
	config.putValue<bool>("Svm.SequentialGamma.RandomNumberGenerator.IsSeedRandom", false);
	config.putValue<int>("Svm.SequentialGamma.RandomNumberGenerator.Seed", 0);

	// @wdudzik Stop condition
	config.putValue<std::string>("Svm.SequentialGamma.StopCondition.Name", "MeanFitness");
	config.putValue<double>("Svm.SequentialGamma.StopCondition.MeanFitness.Epsilon", 1e-3);

	// @wdudzik Selection
	auto truncationCoefficient = 1 / (2 + superIndividualAlpha);
	config.putValue<std::string>("Svm.SequentialGamma.SelectionOperator.Name", "TruncationSelection");
	config.putValue<double>("Svm.SequentialGamma.SelectionOperator.TruncationSelection.TruncationCoefficient", truncationCoefficient);

	// @wdudzik Population generation
	config.putValue<std::string>("Svm.SequentialGamma.Generation.Name", "Random"); //NewKernel - RBF_LINEAR

	// @wdudzik Crossover selection
	config.putValue<std::string>("Svm.SequentialGamma.CrossoverSelection.Name", "LocalGlobalSelection");
	config.putValue<double>("Svm.SequentialGamma.CrossoverSelection.LocalGlobalSelection.HighLowCoefficient", 0.5);
	config.putValue<bool>("Svm.SequentialGamma.CrossoverSelection.LocalGlobalSelection.IsLocalMode", false);

	// @wdudzik Crossover
	config.putValue<std::string>("Svm.SequentialGamma.Crossover.Name", "Memetic");

	// @wdudzik Mutation
	config.putValue<std::string>("Svm.SequentialGamma.Mutation.Name", "GaSvm");
	config.putValue<double>("Svm.SequentialGamma.Mutation.GaSvm.ExchangePercent", 0.3);
	config.putValue<double>("Svm.SequentialGamma.Mutation.GaSvm.MutationProbability", 0.3);

	return config;
}

platform::Subtree DefaultMultipleGammaMASVMConfig::getDefault()
{
	auto config = DefaultSvmConfig::getDefault();

	config.putValue("Name", "MultipleGammaMASVM");

	config.putValue<bool>("Svm.MultipleGammaMASVM.Svm.isRegression", false);


    config.putValue<bool>("Svm.MultipleGammaMASVM.TrainAlpha", true);
    config.putValue<std::string>("Svm.MultipleGammaMASVM.KernelType", "RBF_CUSTOM");
	config.putValue<unsigned int>("Svm.MultipleGammaMASVM.PopulationSize", 10);
	config.putValue<unsigned int>("Svm.MultipleGammaMASVM.NumberOfClasses", 2);
	config.putValue<unsigned int>("Svm.MultipleGammaMASVM.NumberOfClassExamples", 8);

	config.putValue<double>("Svm.MultipleGammaMASVM.Kernel.Gamma", 1.0);
	config.putValue<double>("Svm.MultipleGammaMASVM.Kernel.C", 1.0);


    config.putValue<std::string>("Svm.MultipleGammaMASVM.Validation.Name", "Regular");
	// @wdudzik Memetic specific values
	auto superIndividualAlpha = 0.2;
	config.putValue<double>("Svm.MultipleGammaMASVM.SuperIndividualsAlpha", superIndividualAlpha);
	config.putValue<double>("Svm.MultipleGammaMASVM.PercentOfSupportVectorsThreshold", 0.3);
	config.putValue<unsigned int>("Svm.MultipleGammaMASVM.IterationsBeforeModeChange", 3);
	config.putValue<double>("Svm.MultipleGammaMASVM.EducationProbability", 0.3);
	config.putValue<double>("Svm.MultipleGammaMASVM.ThresholdForMaxNumberOfClassExamples", 1.0);

	// @wdudzik Random number generator
	config.putValue<std::string>("Svm.MultipleGammaMASVM.RandomNumberGenerator.Name", "Mt_19937");
	config.putValue<bool>("Svm.MultipleGammaMASVM.RandomNumberGenerator.IsSeedRandom", false);
	config.putValue<int>("Svm.MultipleGammaMASVM.RandomNumberGenerator.Seed", 0);

	// @wdudzik Stop condition
	config.putValue<std::string>("Svm.MultipleGammaMASVM.StopCondition.Name", "MeanFitness");
	config.putValue<double>("Svm.MultipleGammaMASVM.StopCondition.MeanFitness.Epsilon", 1e-3);

	// @wdudzik Selection
	auto truncationCoefficient = 1 / (2 + superIndividualAlpha);
	config.putValue<std::string>("Svm.MultipleGammaMASVM.SelectionOperator.Name", "TruncationSelection");
	config.putValue<double>("Svm.MultipleGammaMASVM.SelectionOperator.TruncationSelection.TruncationCoefficient", truncationCoefficient);

	// @wdudzik Population generation
	//config.putValue<std::string>("Svm.SequentialGamma.Generation.Name", "Random");

	// @wdudzik Crossover selection
	config.putValue<std::string>("Svm.MultipleGammaMASVM.CrossoverSelection.Name", "LocalGlobalSelection");
	config.putValue<double>("Svm.MultipleGammaMASVM.CrossoverSelection.LocalGlobalSelection.HighLowCoefficient", 0.5);
	config.putValue<bool>("Svm.MultipleGammaMASVM.CrossoverSelection.LocalGlobalSelection.IsLocalMode", false);

	// @wdudzik Crossover
	config.putValue<std::string>("Svm.MultipleGammaMASVM.Crossover.Name", "Memetic");

	// @wdudzik Mutation
	config.putValue<std::string>("Svm.MultipleGammaMASVM.Mutation.Name", "GaSvm");
	config.putValue<double>("Svm.MultipleGammaMASVM.Mutation.GaSvm.ExchangePercent", 0.3);
	config.putValue<double>("Svm.MultipleGammaMASVM.Mutation.GaSvm.MutationProbability", 0.3);

	return config;
}

platform::Subtree DefaultRbfLinearConfig::getDefault()
{
    auto config = DefaultSvmConfig::getDefault();

    config.putValue("Name", "RbfLinear");

    config.putValue<bool>("Svm.RbfLinear.Svm.isRegression", false);


    config.putValue<bool>("Svm.RbfLinear.TrainAlpha", true);

    config.putValue<std::string>("Svm.RbfLinear.KernelType", "RBF_LINEAR");
    config.putValue<unsigned int>("Svm.RbfLinear.NumberOfClassExamples", 2);


    config.putValue<unsigned int>("Svm.RbfLinear.PopulationSize", 10);
    config.putValue<unsigned int>("Svm.RbfLinear.NumberOfClasses", 2);

    config.putValue<double>("Svm.RbfLinear.Kernel.Gamma", 1.0);
    config.putValue<double>("Svm.RbfLinear.Kernel.C", 1.0);
    config.putValue<double>("Svm.RbfLinear.Kernel.Coef0", 1.0);
    config.putValue<double>("Svm.RbfLinear.Kernel.Degree", 3.0);
	
    config.putValue<std::string>("Svm.RbfLinear.Validation.Name", "Regular");

    // @wdudzik Memetic specific values
    auto superIndividualAlpha = 0.2;
    config.putValue<double>("Svm.RbfLinear.SuperIndividualsAlpha", superIndividualAlpha);
    config.putValue<double>("Svm.RbfLinear.PercentOfSupportVectorsThreshold", 0.3);
    config.putValue<unsigned int>("Svm.RbfLinear.IterationsBeforeModeChange", 3);
    config.putValue<double>("Svm.RbfLinear.EducationProbability", 0.3);
    config.putValue<double>("Svm.RbfLinear.ThresholdForMaxNumberOfClassExamples", 1.0);

    // @wdudzik Random number generator
    config.putValue<std::string>("Svm.RbfLinear.RandomNumberGenerator.Name", "Mt_19937");
    config.putValue<bool>("Svm.RbfLinear.RandomNumberGenerator.IsSeedRandom", false);
    config.putValue<int>("Svm.RbfLinear.RandomNumberGenerator.Seed", 0);

    // @wdudzik Stop condition
    config.putValue<std::string>("Svm.RbfLinear.StopCondition.Name", "MeanFitness");
    config.putValue<double>("Svm.RbfLinear.StopCondition.MeanFitness.Epsilon", 1e-3);

    // @wdudzik Selection
    auto truncationCoefficient = 1 / (2 + superIndividualAlpha);
    config.putValue<std::string>("Svm.RbfLinear.SelectionOperator.Name", "TruncationSelection");
    config.putValue<double>("Svm.RbfLinear.SelectionOperator.TruncationSelection.TruncationCoefficient", truncationCoefficient);

    // @wdudzik Population generation
    config.putValue<std::string>("Svm.RbfLinear.Generation.Name", "NewKernel"); 

    // @wdudzik Crossover selection
    config.putValue<std::string>("Svm.RbfLinear.CrossoverSelection.Name", "LocalGlobalSelection");
    config.putValue<double>("Svm.RbfLinear.CrossoverSelection.LocalGlobalSelection.HighLowCoefficient", 0.5);
    config.putValue<bool>("Svm.RbfLinear.CrossoverSelection.LocalGlobalSelection.IsLocalMode", false);

    // @wdudzik Crossover
    config.putValue<std::string>("Svm.RbfLinear.Crossover.Name", "Memetic");

    // @wdudzik Mutation
    config.putValue<std::string>("Svm.RbfLinear.Mutation.Name", "GaSvm");
    config.putValue<double>("Svm.RbfLinear.Mutation.GaSvm.ExchangePercent", 0.3);
    config.putValue<double>("Svm.RbfLinear.Mutation.GaSvm.MutationProbability", 0.3);

    return config;
}

platform::Subtree DefaultSequentialGammaWithFeatureSelectionConfig::getDefault()
{
    auto config = DefaultSvmConfig::getDefault();

    config.putValue("Name", "SequentialGammaFS");

    config.putValue<bool>("Svm.SequentialGamma.Svm.isRegression", false);


    config.putValue<bool>("Svm.SequentialGamma.TrainAlpha", true);
    config.putValue<std::string>("Svm.SequentialGamma.KernelType", "RBF_CUSTOM");
    config.putValue<bool>("Svm.SequentialGamma.ShrinkOnBestOnly", false); //wether shrinking of T should be done based on population or best individual


    config.putValue<unsigned int>("Svm.SequentialGamma.PopulationSize", 10);
    config.putValue<unsigned int>("Svm.SequentialGamma.NumberOfClasses", 2);
    config.putValue<unsigned int>("Svm.SequentialGamma.NumberOfClassExamples", 8);

    config.putValue<double>("Svm.SequentialGamma.Kernel.Gamma", 1.0);
    config.putValue<double>("Svm.SequentialGamma.Kernel.C", 1.0);
    config.putValue<double>("Svm.SequentialGamma.Kernel.Coef0", 1.0);
    config.putValue<double>("Svm.SequentialGamma.Kernel.Degree", 3.0);


    config.putValue<std::string>("Svm.SequentialGamma.Validation.Name", "Regular");


    // @wdudzik Memetic specific values
    auto superIndividualAlpha = 0.2;
    config.putValue<double>("Svm.SequentialGamma.SuperIndividualsAlpha", superIndividualAlpha);
    config.putValue<double>("Svm.SequentialGamma.PercentOfSupportVectorsThreshold", 0.3);
    config.putValue<unsigned int>("Svm.SequentialGamma.IterationsBeforeModeChange", 3);
    config.putValue<double>("Svm.SequentialGamma.EducationProbability", 0.3);
    config.putValue<double>("Svm.SequentialGamma.ThresholdForMaxNumberOfClassExamples", 1.0);

    // @wdudzik Random number generator
    config.putValue<std::string>("Svm.SequentialGamma.RandomNumberGenerator.Name", "Mt_19937");
    config.putValue<bool>("Svm.SequentialGamma.RandomNumberGenerator.IsSeedRandom", false);
    config.putValue<int>("Svm.SequentialGamma.RandomNumberGenerator.Seed", 0);

    // @wdudzik Stop condition
    config.putValue<std::string>("Svm.SequentialGamma.StopCondition.Name", "MeanFitness");
    config.putValue<double>("Svm.SequentialGamma.StopCondition.MeanFitness.Epsilon", 1e-3);

    // @wdudzik Selection
    auto truncationCoefficient = 1 / (2 + superIndividualAlpha);
    config.putValue<std::string>("Svm.SequentialGamma.SelectionOperator.Name", "TruncationSelection");
    config.putValue<double>("Svm.SequentialGamma.SelectionOperator.TruncationSelection.TruncationCoefficient", truncationCoefficient);

    // @wdudzik Population generation
    config.putValue<std::string>("Svm.SequentialGamma.Generation.Name", "Random"); //NewKernel - RBF_LINEAR

    // @wdudzik Crossover selection
    config.putValue<std::string>("Svm.SequentialGamma.CrossoverSelection.Name", "LocalGlobalSelection");
    config.putValue<double>("Svm.SequentialGamma.CrossoverSelection.LocalGlobalSelection.HighLowCoefficient", 0.5);
    config.putValue<bool>("Svm.SequentialGamma.CrossoverSelection.LocalGlobalSelection.IsLocalMode", false);

    // @wdudzik Crossover
    config.putValue<std::string>("Svm.SequentialGamma.Crossover.Name", "Memetic");

    // @wdudzik Mutation
    config.putValue<std::string>("Svm.SequentialGamma.Mutation.Name", "GaSvm");
    config.putValue<double>("Svm.SequentialGamma.Mutation.GaSvm.ExchangePercent", 0.3);
    config.putValue<double>("Svm.SequentialGamma.Mutation.GaSvm.MutationProbability", 0.3);


    config.putNode("Svm.MemeticFeatureSetSelection", DefaultFeaturesMemeticConfig::getDefault().getNode("Svm.MemeticFeatureSetSelection"));
    // @wdudzik Choose which algorithm will run in KTF
    config.putValue("Svm.seq.FeatureSetOptimization.Name", "MemeticFeatureSetSelection");


    
    return config;
}


platform::Subtree DefaultEnsembleConfig::getDefault()
{
    auto config = DefaultSvmConfig::getDefault();

    config.putValue("Name", "Ensemble");
    config.putValue<bool>("Svm.Alga.Svm.isRegression", false);

    // @wdudzik Insert default configs for algorithms used in Alga
    config.putNode("Svm.MemeticTrainingSetSelection", DefaultMemeticConfig::getDefault().getNode("Svm.MemeticTrainingSetSelection"));
    config.putNode("Svm.GeneticKernelEvolution", DefaultKernelEvolutionConfig::getDefault().getNode("Svm.GeneticKernelEvolution"));

    // @wdudzik Choose which algorithm will run in Alga
    config.putValue("Svm.Alga.KernelOptimization.Name", "GeneticKernelEvolution");
    config.putValue("Svm.Alga.TrainingSetOptimization.Name", "MemeticTrainingSetSelection");

    return config;
}

platform::Subtree DefaultEnsembleTreeConfig::getDefault()
{
    auto config = DefaultSvmConfig::getDefault();

    config.putValue("Name", "EnsembleTree");
    config.putValue<bool>("Svm.EnsembleTree.Svm.isRegression", false);

    // @wdudzik Insert default configs for algorithms used in KTF
    config.putNode("Svm.MemeticTrainingSetSelection", DefaultMemeticConfig::getDefault().getNode("Svm.MemeticTrainingSetSelection"));
    config.putNode("Svm.GeneticKernelEvolution", DefaultKernelEvolutionConfig::getDefault().getNode("Svm.GeneticKernelEvolution"));
    config.putNode("Svm.MemeticFeatureSetSelection", DefaultFeaturesMemeticConfig::getDefault().getNode("Svm.MemeticFeatureSetSelection"));

	// for ensemble we setup mak k value 
    config.putValue<double>("Svm.MemeticTrainingSetSelection.Memetic.MaxK", 0); //8
	
    // @wdudzik Choose which algorithm will run in KTF
    config.putValue("Svm.EnsembleTree.KernelOptimization.Name", "GeneticKernelEvolution");
    config.putValue("Svm.EnsembleTree.TrainingSetOptimization.Name", "MemeticTrainingSetSelection");
    config.putValue("Svm.EnsembleTree.FeatureSetOptimization.Name", "MemeticFeatureSetSelection");



    //this depends on the size of Memetic Feature set and memetic training set
    auto superIndividualAlpha = 0.2;
    //auto truncationCoefficient = 2 / (2*2 + superIndividualAlpha);
    auto truncationCoefficient = 1 / (2 + superIndividualAlpha);
    config.putValue("Svm.EnsembleTree.SelectionOperator.Name", "TruncationSelection");
    config.putValue<double>("Svm.EnsembleTree.SelectionOperator.TruncationSelection.TruncationCoefficient", truncationCoefficient);
    config.putValue("Svm.EnsembleTree.PopulationSize", 10);

    config.putValue<bool>("Svm.EnsembleTree.ConstKernel", false); //Note this may changed kernel used
    config.putValue<bool>("Svm.EnsembleTree.SwitchFitness", false);

    config.putValue<bool>("Svm.EnsembleTree.AddSvToTraining", false); //wheter to insert sv into population or just leave them at training set based on SvMode
    config.putValue<std::string>("Svm.EnsembleTree.SvMode", "defaultOption"); //default is none so no sv adding

    config.putValue<std::string>("Svm.EnsembleTree.GrowKMode", "defaultOption");

	config.putValue<bool>("Svm.EnsembleTree.DasvmKernel", false);
    config.putValue<bool>("Svm.EnsembleTree.DebugLog", false);
    config.putValue<bool>("Svm.EnsembleTree.AddAlmaNode", false);
    config.putValue<bool>("Svm.EnsembleTree.UseFeatureSelction", false);

    config.putValue<bool>("Svm.EnsembleTree.UseImbalanceRatio", false);

    config.putValue<bool>("Svm.EnsembleTree.NewDatasetSampling", false);

    config.putValue<bool>("Svm.EnsembleTree.NewSamplesForTraining", false);
    config.putValue<bool>("Svm.EnsembleTree.ResamplingWithNoAddition", false);


    config.putValue<bool>("Svm.EnsembleTree.UseSingleClassThresholds", false);
    config.putValue<bool>("Svm.EnsembleTree.UseBias", false);
    config.putValue<bool>("Svm.EnsembleTree.NewFlowFullValidationSet", false);


	
    return config;
}

platform::Subtree DefaultSESVMCorrectedConfig::getDefault()
{
    auto config = DefaultSvmConfig::getDefault();

    config.putValue("Name", "SESVM_Corrected");
    config.putValue<bool>("Svm.SSVM.Svm.isRegression", false);

    // @wdudzik Insert default configs for algorithms used in KTF
    config.putNode("Svm.MemeticTrainingSetSelection", DefaultMemeticConfig::getDefault().getNode("Svm.MemeticTrainingSetSelection"));
    config.putNode("Svm.GeneticKernelEvolution", DefaultKernelEvolutionConfig::getDefault().getNode("Svm.GeneticKernelEvolution"));
    config.putNode("Svm.MemeticFeatureSetSelection", DefaultFeaturesMemeticConfig::getDefault().getNode("Svm.MemeticFeatureSetSelection"));

    // @wdudzik Choose which algorithm will run in KTF
    config.putValue("Svm.SSVM.KernelOptimization.Name", "GeneticKernelEvolution");
    config.putValue("Svm.SSVM.TrainingSetOptimization.Name", "MemeticTrainingSetSelection");
    config.putValue("Svm.SSVM.FeatureSetOptimization.Name", "MemeticFeatureSetSelection");



    //this depends on the size of Memetic Feature set and memetic training set
    auto superIndividualAlpha = 0.1;
    auto truncationCoefficient = 1 / (2 + superIndividualAlpha);
    config.putValue("Svm.SSVM.SelectionOperator.Name", "TruncationSelection");
    config.putValue<double>("Svm.SSVM.SelectionOperator.TruncationSelection.TruncationCoefficient", truncationCoefficient);
    config.putValue("Svm.SSVM.PopulationSize", 20);

    return config;
}

platform::Subtree DefaultBigSetsEnsembleConfig::getDefault()
{
    auto config = DefaultEnsembleTreeConfig::getDefault();

    config.putValue("Name", "BigSetsEnsemble");


    config.putValue<bool>("Svm.EnsembleTree.UseSingleClassThresholds", true);
    config.putValue<bool>("Svm.EnsembleTree.UseBias", true);
    config.putValue<bool>("Svm.EnsembleTree.NewFlowFullValidationSet", false);
    //config.putValue<bool>("Svm.EnsembleTree.AddSvToTraining", false);
    //config.putValue<std::string>("Svm.EnsembleTree.SvMode", "global");
    config.putValue<std::string>("Svm.EnsembleTree.GrowKMode", "zeroOut");
    //config.putValue<bool>("Svm.EnsembleTree.DasvmKernel", true);
    //config.putValue<bool>("Svm.EnsembleTree.AddAlmaNode", true);
    //config.putValue<bool>("Svm.EnsembleTree.UseFeatureSelction", true);
    
    // for ensemble we setup mak k value 
    //config.putValue<double>("Svm.MemeticTrainingSetSelection.Memetic.MaxK", 128); //8
    config.putValue<double>("Svm.MemeticTrainingSetSelection.Memetic.MaxK", 0); //8

    config.putValue<bool>("Svm.EnsembleTree.UseFeatureSelctionCascadeWise", false);

	
    return config;
}
} // namespace genetic
