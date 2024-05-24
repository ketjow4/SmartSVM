

#pragma warning(disable : 4244)
#pragma warning(disable : 4996)

#include <cmath>

#include <libPlatform/Subtree.h>
#include "libGeneticSvm/DefaultWorkflowConfigs.h"

//#include <TestApp/RunAlgorithm.h>
#include <libRandom/MersenneTwister64Rng.h>
#include <SvmLib/libSvmImplementation.h>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/iostream.h>

#include <libPlatform/loguru.cpp>

#include <libDataset/Dataset.h>

#include "DatasetLoader.h"
#include "libGeneticSvm/IDatasetLoader.h"
#include "libGeneticSvm/LocalFileDatasetLoader.h"
#include "libGeneticSvm/SvmAlgorithmFactory.h"

namespace py = pybind11;
using namespace platform;


std::vector<unsigned int> countLabels(unsigned int numberOfClasses, const dataset::Dataset<std::vector<float>, float>& dataset)
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


void print_10(dataset::Dataset<std::vector<float>, float> dataset)
{
    for (auto i = 0u; i < 10; i++)
    {
        std::cout << "Sample: ";
        const auto sample = dataset.getSample(i);
        for (auto j = 0u; j < sample.size(); j++)
        {
            std::cout << sample[j] << ", ";
        }

        std::cout << "  Label: " << dataset.getLabel(i) << "\n";
    }
}



class GeneticSvm
{
public:
	virtual ~GeneticSvm() = default;


	virtual void fit(py::array_t<float> tr_x, py::array_t<float> tr_y,
	                 py::array_t<float> val_x, py::array_t<float> val_y,
	                 py::array_t<float> test_x, py::array_t<float> test_y)
    {
        try
        {
            std::unique_ptr<genetic::IDatasetLoader> dataLoader = std::make_unique<NumpyDatasetLoader>(tr_x, tr_y, val_x, val_y, test_x, test_y);


            auto algorithm = genetic::SvmAlgorithmFactory::createAlgorightm(m_config, *dataLoader);

            std::cout << "Created algorithm" << std::endl;

            m_trainedModel = algorithm->run();

            std::cout << "Finished algorithm" << std::endl;
        }
        catch (std::runtime_error& e)
        {
            std::cout << e.what();
        }
    }


    virtual std::vector<float> predict(py::array_t<float> samples) const
    {
        py::buffer_info bufInfo = samples.request();
        std::vector<float> answers;

        if (bufInfo.ndim == 2)
        {
            auto manySamples = numpyToVectorOfVectors(samples);
            answers.reserve(manySamples.size());


            for (auto& sample : manySamples)
            {
                answers.emplace_back(m_trainedModel->classify(sample));
            }
        }
        else if (bufInfo.ndim == 1)
        {
            auto sample = numpyToVector(samples);
            answers.emplace_back(m_trainedModel->classify(sample));
        }
        else
        {
            throw std::runtime_error("Input array must be 1-dimensional or 2-dimensional");
        }

        return answers;
    }

    std::shared_ptr<phd::svm::ISvm> get()
    {
        return m_trainedModel;
    }


    Subtree m_config;

private:
    std::shared_ptr<phd::svm::ISvm> m_trainedModel;

};


class ALMA_python : public GeneticSvm
{
public:
    //ALMA_python( verbosity, outputFolder, paramDict)
	ALMA_python()
	{
        std::cout << "Called constructor of ALMA" << std::endl;
        loguru::g_stderr_verbosity = loguru::Verbosity_INFO;
        loguru::add_file("ALMA.log", loguru::Append, loguru::Verbosity_MAX);
        loguru::flush();

		m_config = genetic::DefaultAlgaConfig::getALMA();
	}
};


class SESVM_python : public GeneticSvm
{
public:
    SESVM_python()
    {
        std::cout << "Called constructor of SESVM_python" << std::endl;
        loguru::g_stderr_verbosity = loguru::Verbosity_INFO;
        loguru::add_file("SESVM.log", loguru::Append, loguru::Verbosity_MAX);

        loguru::flush();

        m_config = genetic::DefaultSESVMConfig::getDefault();
    }
};

class GASVM_python : public GeneticSvm
{
public:
    GASVM_python(double C, double gamma, phd::svm::KernelTypes kernel)
    {
        std::cout << "Called constructor of GASVM_python" << std::endl;
        loguru::g_stderr_verbosity = loguru::Verbosity_INFO;
        loguru::add_file("GASVM.log", loguru::Append, loguru::Verbosity_MAX);
        loguru::flush();

        m_config = genetic::DefaultGaSvmConfig::getDefault();
        m_config.putValue<double>("Svm.GaSvm.Kernel.Gamma", gamma);
        m_config.putValue<double>("Svm.GaSvm.Kernel.C", C);
        m_config.putValue("Svm.KernelType", "RBF");
    }
};


class MASVM_python : public GeneticSvm
{
public:
    MASVM_python(double C, double gamma, phd::svm::KernelTypes kernel)
    {
        std::cout << "Called constructor of MASVM_python" << std::endl;
        loguru::g_stderr_verbosity = loguru::Verbosity_INFO;
        loguru::add_file("MASVM.log", loguru::Append, loguru::Verbosity_MAX);
        loguru::flush();

        m_config = genetic::DefaultMemeticConfig::getDefault();
        m_config.putValue<double>("Svm.MemeticTrainingSetSelection.Kernel.Gamma", gamma);
        m_config.putValue<double>("Svm.MemeticTrainingSetSelection.Kernel.C", C);
        m_config.putValue("Svm.KernelType", "RBF");
    }
};


class CESVM_python : public GeneticSvm
{
public:
    CESVM_python()
    {
        std::cout << "Called constructor of CESVM_python" << std::endl;
        loguru::g_stderr_verbosity = loguru::Verbosity_INFO;
        loguru::add_file("CESVM.log", loguru::Append, loguru::Verbosity_MAX);
        loguru::flush();

        // GECCO 2022 config
        m_config = genetic::DefaultEnsembleTreeConfig::getDefault();
        m_config.putValue<bool>("Svm.EnsembleTree.AddSvToTraining", true);
        m_config.putValue<std::string>("Svm.EnsembleTree.SvMode", "global");
        m_config.putValue<std::string>("Svm.EnsembleTree.GrowKMode", "zeroOut");
        m_config.putValue<bool>("Svm.EnsembleTree.DasvmKernel", true);
        m_config.putValue<bool>("Svm.EnsembleTree.AddAlmaNode", true);
    }
};


class ECESVM_python : public GeneticSvm
{
public:
    ECESVM_python()
    {
        std::cout << "Called constructor of ECESVM_python" << std::endl;
        loguru::g_stderr_verbosity = loguru::Verbosity_INFO;
        loguru::add_file("ECESVM.log", loguru::Append, loguru::Verbosity_MAX);
        loguru::flush();

        // ECE-SVM knosys config
        m_config = genetic::DefaultBigSetsEnsembleConfig::getDefault();

        //TODO add Extra tree as a last node and create nice classifier
    }
};


class DASVM_CE_NOSMO_python : public GeneticSvm
{
public:
    DASVM_CE_NOSMO_python()
    {
        std::cout << "Called constructor of DASVM_CE_NOSMO_python" << std::endl;
        loguru::g_stderr_verbosity = loguru::Verbosity_INFO;
        loguru::add_file("DASVM_CE_NOSMO.log", loguru::Append, loguru::Verbosity_MAX);
        loguru::flush();

        m_config = genetic::DefaultRbfLinearConfig::getDefault();
        //rbfLinearCoevolution_nosmo.putValue<unsigned int>("Svm.RbfLinear.NumberOfClassExamples", info.numberOfFeatures);
        m_config.putValue<std::string>("Svm.RbfLinear.SelectionOperator.Name", "ConstatntTruncationSelection");
        m_config.putValue("Name", "RbfLinearCoevolution");
        m_config.putValue<bool>("Svm.RbfLinear.TrainAlpha", false);

        m_config.putValue("Name", "RbfLinearCoevolutionCENOT");
        //auto p = std::pair<std::string, platform::Subtree>{ "DA_SVM_CE_NO_T",  rbfLinearCoevolution_nosmo };
    }
};

PYBIND11_MODULE(DeevaPythonPackage, m) {

    // Redirect C++ std::cout to Python sys.stdout
    py::scoped_ostream_redirect output;
    py::scoped_estream_redirect error_stream;

    m.def("print10", &print_10, R"pbdoc(
        Prints first 10 rows of dataset to check if everything is correctly converted between C++ and Python
    )pbdoc");

    m.def("countLabels", &countLabels, py::arg("numberOfClasses"), py::arg("dataset"), R"pbdoc(
        count lables in dataset in C++ format
    )pbdoc");

    m.def("convertToDataset", &convertToDataset, py::arg("data"), py::arg("labels"), R"pbdoc(
        converts numpy array to C++ dataset
    )pbdoc");

    py::class_<GeneticSvm, std::shared_ptr<GeneticSvm>>(m, "GeneticSvm")
        .def("fit", &GeneticSvm::fit, py::arg("TrX"), py::arg("TrY"), py::arg("ValX"), py::arg("ValY"), py::arg("TestX"), py::arg("TestY"), R"pbdoc(
        Runs ALMA algorithm and fits the SVM model. Test data is used only for logging results on the test set
		)pbdoc")
        .def("predict", &GeneticSvm::predict, py::arg("samples"), R"pbdoc(
        Runs predcition on already trained model. If fit was not called previously it thorws error. Handle 1-D and 2-D numpy arrays as inputs (each row is a sample)
		)pbdoc")
        .def_property("config",
            [](GeneticSvm& self)
            {
                return self.m_config;
            },
            [](GeneticSvm& self, Subtree config)
            {
                self.m_config = config;
            })
        .def_property_readonly("svm", &GeneticSvm::get);

    py::class_<ALMA_python, GeneticSvm, std::shared_ptr<ALMA_python>> alma(m, "AlmaClassifier", R"pbdoc(
        TODO:
     )pbdoc");
    alma
        .def(py::init<>());


    py::class_<SESVM_python, GeneticSvm, std::shared_ptr<SESVM_python>> sesvm(m, "SeSvmClassifier", R"pbdoc(
       TODO:
     )pbdoc");
    sesvm
        .def(py::init<>());


    py::class_<CESVM_python, GeneticSvm, std::shared_ptr<CESVM_python>> cesvm(m, "CeSvmClassifier", R"pbdoc(
       TODO:
     )pbdoc");
    cesvm
        .def(py::init<>());

    py::class_<ECESVM_python, GeneticSvm, std::shared_ptr<ECESVM_python>> ecesvm(m, "EceSvmClassifier", R"pbdoc(
       TODO:
     )pbdoc");
    ecesvm
        .def(py::init<>());


    py::class_<MASVM_python, GeneticSvm, std::shared_ptr<MASVM_python>> masvm(m, "MaSvmClassifier", R"pbdoc(
       TODO:
     )pbdoc");
    masvm
        .def(py::init<double, double, phd::svm::KernelTypes>(), py::arg("C"), py::arg("gamma"), py::arg("kernelType"));

    py::class_<GASVM_python, GeneticSvm, std::shared_ptr<GASVM_python>> gasvm(m, "GaSvmClassifier", R"pbdoc(
       TODO:
     )pbdoc");
    gasvm
        .def(py::init<double, double, phd::svm::KernelTypes>(), py::arg("C"), py::arg("gamma"), py::arg("kernelType"));
       
      


    py::class_<dataset::Dataset<std::vector<float>, float>> dataset(m, "Dataset", R"pbdoc(
        Class that handles dataset in C++
     )pbdoc");


    dataset.def_property_readonly_static("labels", [](dataset::Dataset<std::vector<float>, float>& self) {
    	return self.getLabels(); //TODO fix types as this returns span which is incompatible
        });

    py::class_<Subtree> subtree(m, "Subtree", R"pbdoc(
        Class that handles all of the configurations, reads from and seralize to JSON format
     )pbdoc");
    subtree
        .def(py::init<const std::filesystem::path&>())
        .def(py::init<std::string>(), pybind11::arg("jsonString"))
        .def("save", &Subtree::save)
        //.def("putValue", &Subtree::putValue<std::string>)
        .def("putValue", [](Subtree& self, const std::string& name, const std::string& value)
            { return self.putValue<std::string>(name, value); })
        .def("putValue", [](Subtree& self, const std::string& name, const double& value)
            { return self.putValue<double>(name, value); })
        .def("putValue", [](Subtree& self, const std::string& name, const bool& value)
        { return self.putValue<bool>(name, value); })
        .def("putValue", [](Subtree& self, const std::string& name, const long& value)
        { return self.putValue<long>(name, value); })
        //.def("putValue", &Subtree::putValue<long>)
        //.def("putValue", &Subtree::putValue<double>)
        .def("to_string", &Subtree::writeToString)
        .def("__str__", [](const Subtree& s) {return s.writeToString(); });

    py::class_<genetic::DefaultAlgaConfig> algaConfig(m, "AlgaConfig", R"pbdoc(
        Class that hold default configuration for algorithm
     )pbdoc");

    algaConfig.def("getALMA", &genetic::DefaultAlgaConfig::getALMA);


    py::enum_<phd::svm::KernelTypes>(m, "KernelType")
        .value("Rbf", phd::svm::KernelTypes::Rbf)
        .value("Linear", phd::svm::KernelTypes::Linear)
        .value("Polynomial", phd::svm::KernelTypes::Poly);
  

#ifdef VERSION_INFO
    m.attr("__version__") = VERSION_INFO;
#else
    m.attr("__version__") = "0.3.9";
#endif
}