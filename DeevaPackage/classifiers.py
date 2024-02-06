
from DeevaPythonPackage import  AlmaClassifier



def alma_wrapper(dataset):
    algorithm = AlmaClassifier()
    config = algorithm.config

    config.putValue("Svm.MemeticTrainingSetSelection.StopCondition.MeanFitness.Epsilon", 0.001)
    config.putValue("Svm.GeneticKernelEvolution.StopCondition.MeanFitness.Epsilon", 0.001)
    config.putValue("Svm.Metric", "AUC")
    config.putValue("Svm.MemeticFeatureSetSelection.NumberOfClassExamples", 2)
    algorithm.config = config

    print(config)

    algorithm.fit(dataset.X_tr, dataset.Y_tr, dataset.X_val, dataset.Y_val, dataset.X_test, dataset.Y_test)

    responses = algorithm.predict(dataset.X_test)

    #cm = get_confusion_matrix(dataset, algorithm)
    print(f'Run success')


def hello():
    print("hello world")
    



