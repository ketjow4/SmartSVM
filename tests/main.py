from sklearn.datasets import load_iris 
import numpy as np

import os

print(os.getcwd())

import DeevaPythonPackage
from DeevaPythonPackage import  Subtree, AlmaClassifier, SeSvmClassifier, MaSvmClassifier, GaSvmClassifier, KernelType, CeSvmClassifier, EceSvmClassifier

from DeevaPackage.classifiers import alma_wrapper

# from BaselineExperiments.commons import Dataset, load_all_5_folds_T_equal_V, get_confusion_matrix, scaleData
from commons import Dataset, load_all_5_folds_T_equal_V, scaleData

if __name__ == "__main__":
    # hello()

    print(f'Package version {DeevaPythonPackage.__version__}')
    
    # print(fast_tanh(10))
    # print(f'Random from C++ " {getRandom()}')
    # print(f'Get default C svm {DeevaPythonPackage.someSvmFunction()}')

    s = DeevaPythonPackage.AlgaConfig.getALMA()

    string_json = s.to_string()


    #load csv, divide into X and Y, pass to C++ --> count labels + print first 10 rows
    # X, Y = data = load_iris(return_X_y=True)

    # cpp_dataset = DeevaPythonPackage.convertToDataset(X,Y)

    # count = DeevaPythonPackage.countLabels(3,cpp_dataset)

    # print(count)

    print(f'Training SVM with ALMA example')
    traningSetPath = r'E:\PythonDeevaTests\2D_shapes'

    dataset = load_all_5_folds_T_equal_V(traningSetPath)[0]

    cpp2D_dataset = DeevaPythonPackage.convertToDataset(np.ascontiguousarray(dataset.X_tr), dataset.Y_tr)

    DeevaPythonPackage.print10(cpp2D_dataset)

    # data = Dataset(np.ascontiguousarray(dataset.X_tr), dataset.Y_tr, 
    #                np.ascontiguousarray(dataset.X_val), dataset.Y_val, 
    #                np.ascontiguousarray(dataset.X_test), dataset.Y_test)

    data = Dataset(np.ascontiguousarray(dataset.X_tr), dataset.Y_tr, 
                np.ascontiguousarray(dataset.X_val), dataset.Y_val, 
                np.ascontiguousarray(np.empty([0,0])), np.empty(0))

    # dataset = scaleData(data)
    dataset = data
    try:
        algorithms = [
            AlmaClassifier(),  
            # MaSvmClassifier(1,1,KernelType.Rbf), 
            # GaSvmClassifier(1,1,KernelType.Rbf),
            # SeSvmClassifier(),  #Change number of Feature to 2
            # CeSvmClassifier(),
            # EceSvmClassifier(),
            ]
                      
        for algorithm in algorithms:
            config = algorithm.config

            config.putValue("Svm.MemeticTrainingSetSelection.StopCondition.MeanFitness.Epsilon", 0.001)
            config.putValue("Svm.GeneticKernelEvolution.StopCondition.MeanFitness.Epsilon", 0.001)
            config.putValue("Svm.Metric", "AUC")
            config.putValue("Svm.MemeticFeatureSetSelection.NumberOfClassExamples", 2)

            # config.putValue("Svm.Visualization.Create", True)

            algorithm.config = config

            print(config)

            algorithm.fit(dataset.X_tr, dataset.Y_tr, dataset.X_val, dataset.Y_val, dataset.X_test, dataset.Y_test)

            responses = algorithm.predict(dataset.X_test)

            # cm = get_confusion_matrix(dataset, algorithm)
            print(f'Run success')
    except Exception as exc:
        print(exc)


    alma_wrapper(dataset)







