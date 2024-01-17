#pragma once

#include <map>
#include <filesystem>
//#include <opencv2/ml.hpp>
#include <gsl/span>


#include "libPlatform/loguru.hpp"
#include "ISvm.h"
#include "SvmExceptions.h"
#include "libSvmInternal.h"
//#include "libSvmComponents/IGroupPropagation.h"

namespace phd { namespace svm
{
class libSvmImplementation : public ISvm
{
    static constexpr double Nan = std::numeric_limits<double>::signaling_NaN();
public:
    ~libSvmImplementation() override;
    libSvmImplementation();
    libSvmImplementation(const std::filesystem::path& filepath);

    //libSvmImplementation(std::shared_ptr<svmComponents::IGroupPropagation> strategy);

    //libSvmImplementation(const std::string& model_text);
    std::string saveToString();
    void loadFromString(const std::string& model_text);

    libSvmImplementation(const libSvmImplementation& other) = default;
    explicit libSvmImplementation(const std::filesystem::path& filepath, const dataset::Dataset<std::vector<float>, float>& trainingSet);

    bool isMaxIterReached() const;
    std::vector<int> getSvLables() const;


    double getC() const override;
    double getGamma() const override;
	std::vector<double> getGammas() const override;
    double getCoef0() const override;
    double getDegree() const override;
    double getNu() const override;
    double getP() const override;
    void setC(double value) override;
    void setCoef0(double value) override;
    void setDegree(double value) override;
    void setGamma(double value) override;
    void setGammas(const std::vector<double>& value) override;
    void setNu(double value) override;
    void setP(double value) override;
    KernelTypes getKernelType() const override;
    void setKernel(KernelTypes kernelType) override;
    SvmTypes getType() const override;
    void setType(SvmTypes svmType) override;
    void save(const std::filesystem::path& filepath) override;
    uint32_t getNumberOfKernelParameters(KernelTypes kernelType) const override;
    uint32_t getNumberOfSupportVectors() const override;
    std::vector<std::vector<float>> getSupportVectors() const override;

    float classify(const gsl::span<const float> sample) const override;
    void train(const dataset::Dataset<std::vector<float>, float>& trainingSet, bool probabilityNeeded = false) override;

    double classificationProbability(const gsl::span<const float> sample) const override;
    //void setTerminationCriteria(const cv::TermCriteria& value) override;
    // cv::TermCriteria getTerminationCriteria() const override;
    bool isTrained() const override;
    bool canGiveProbabilityOutput() const override;

    double classifyWithOptimalThreshold(const gsl::span<const float> sample) const override;
    void setOptimalProbabilityThreshold(double optimalThreshold) override;
    bool canClassifyWithOptimalThreshold() const override;

    std::unordered_map<int, int> classifyGroups(const dataset::Dataset<std::vector<float>, float>& dataWithGroups) const override
    {
        std::unordered_map<int, int> answers;
        // auto& groups = dataWithGroups.getGroups();
        // std::unordered_map<int, std::vector<int>> groupsToIndex;

        // for(auto i = 0u; i < groups.size(); ++i)
        // {
        //     groupsToIndex[static_cast<int>(groups[i])].emplace_back(static_cast<int>(i));
        // }

        // auto samples = dataWithGroups.getSamples();
        
        // for(auto& [group, indexes] : groupsToIndex)
        // {
        //     std::vector<float> group_answers;
        //     for(auto& index : indexes)
        //     {
        //         group_answers.emplace_back(classifyHyperplaneDistance(samples[index]));
        //     }

        //     const auto propagatedAnswer = m_groupStrategy->propagateAnswer(group_answers);
        //     auto finalClass = propagatedAnswer > 0 ? 1 : 0;

        //     if (m_model->param.m_optimalThresholdSet)
        //     {
        //         finalClass = propagatedAnswer > m_model->param.m_optimalProbabilityThreshold ? 1 : 0;
        //     }

        //     answers[group] = finalClass;
        // }

        return answers;
    }


    std::unordered_map<int, float> classifyGroupsRawScores(const dataset::Dataset<std::vector<float>, float>& dataWithGroups) const override
    {
        std::unordered_map<int, float> answers;
        // auto& groups = dataWithGroups.getGroups();
        // std::unordered_map<int, std::vector<int>> groupsToIndex;

        // for (auto i = 0u; i < groups.size(); ++i)
        // {
        //     groupsToIndex[static_cast<int>(groups[i])].emplace_back(static_cast<int>(i));
        // }

        // auto samples = dataWithGroups.getSamples();

        // for (auto& [group, indexes] : groupsToIndex)
        // {
        //     std::vector<float> group_answers;
        //     for (auto& index : indexes)
        //     {
        //         group_answers.emplace_back(classifyHyperplaneDistance(samples[index]));
        //     }

        //     const auto propagatedAnswer = m_groupStrategy->propagateAnswer(group_answers);
        //     answers[group] = propagatedAnswer;
        // }

        return answers;
    }

    float classifyDistanceToClosestSV(const gsl::span<const float> sample) const;

    float classifyWithCertainty(const gsl::span<const float> sample) const override;

    void setFeatureSet(const std::vector<svmComponents::Feature>& features, int numberOfFeatures) override;
    const std::vector<svmComponents::Feature>& getFeatureSet() override;

    float classifyHyperplaneDistance(const gsl::span<const float> sample) const override;

    std::tuple<double, double, int>  classifyPositiveNegative(const gsl::span<const float> sample) const;

    //float classifyWithCertainty(const gsl::span<const float> sample) const override;

	std::tuple<std::multimap<int, int>, std::vector<double>> check_sv(const dataset::Dataset<std::vector<float>, float>& validation);

    void setAlphaTraining(bool value);

    double getT() const override;
    void setT(double value) override;

    double getMinSvDistance();  //TODO remove in future

    std::vector<uint64_t> getCertaintyRegion(const dataset::Dataset<std::vector<float>, float>& dataset);
    std::vector<uint64_t> getUncertaintyRegion(const dataset::Dataset<std::vector<float>, float>& dataset);
	
	void setupSupportVectorLoadedFromFile(const dataset::Dataset<std::vector<float>, float>& trainingSet);

    svm_problem createDatasetForTraining(const dataset::Dataset<std::vector<float>, float>& trainingSet);

    void setCertaintyThreshold(double negative, double positive, double normalizedNegative, double normalizedPositive);
    void setClassCertaintyThreshold(double negative, double positive, double normalizedNegative, double normalizedPositive);
    double getPositiveCertainty() const;
    double getNegativeCertainty() const;

    double getPositiveNormalizedCertainty() const;
    double getNegativeNormalizedCertainty() const;

    bool m_isMulticlass;
    svm_parameter m_param;
    svm_model* m_model;
    svm_problem m_problem; //used for traning but need to be free at the destruction if model not loaded from file
    //cv::Mat m_sv;
    std::vector<std::vector<float>> m_sv;
    
	
private:
    std::vector<svm_node> convertSample(gsl::span<const float> sample) const;
	void setupSupportVector(const dataset::Dataset<std::vector<float>, float>& trainingSet);


    svm_problem copy()
    {
        svm_problem problem;

        // Set size of train set.
        problem.l = static_cast<int>(m_problem.l);

        // Set labels of data
        problem.y = new double[problem.l];
        for (int i = 0; i < m_problem.l; i++)
        {
            problem.y[i] = m_problem.y[i];
        }

        unsigned int dims = 0;
        while (m_problem.x[0][dims].index != -1)
        {
            dims++;
        }

        // Set data
       
        problem.x = new svm_node * [problem.l];
        for (int i = 0; i < problem.l; i++)
        {
            problem.x[i] = new svm_node[dims + 1];
            int filledIndex = 0;
            for (unsigned j = 0; j < dims; j++)
            {
                if (m_problem.x[i][j].value != 0)
                {
                    problem.x[i][filledIndex].index = m_problem.x[i][j].index;
                    problem.x[i][filledIndex].value = m_problem.x[i][j].value;
                    filledIndex++;
                }
            }
            problem.x[i][filledIndex].index = -1;
            problem.x[i][filledIndex].value = 0;
        }

        return problem;
    }

    svm_model* copy_model()
    {
        svm_model* model = new svm_model;
        model->param = m_model->param; //check this part ?????
        model->l = m_model->l;
        model->nr_class = m_model->nr_class;
        model->free_sv = 0; //check -- custom extenstion
    	
        auto k = m_model->nr_class;

    	model->rho = new double[k * (k - 1) / 2];
        //std::copy(&m_model->rho, &m_model->rho + (k * (k - 1) / 2), &model->rho);
        std::copy_n(m_model->rho, k * (k - 1) / 2, model->rho);

        model->label = new int[k];
        //std::copy(&m_model->label, &m_model->label + k, &model->label);
        std::copy_n(m_model->label, k, model->label);

        model->nSV = new int[k];
        //std::copy(&m_model->nSV, &m_model->nSV + k, &model->nSV);
        std::copy_n(m_model->nSV, k, model->nSV);

        model->sv_coef = new double* [k-1];
        for (auto i = 0; i < k-1; i++)
        {
             model->sv_coef[i] = new double[m_model->l];
             //std::copy(&m_model->sv_coef[i], &m_model->sv_coef[i] + m_model->l, &model->sv_coef[i]);
             std::copy_n(m_model->sv_coef[i], m_model->l, model->sv_coef[i]);
        }


        unsigned int dims = 0;
        while (m_problem.x[0][dims].index != -1)
        {
            dims++;
        }

        model->sv_indices = new int[m_model->l];
        //std::copy(&m_model->sv_indices, &m_model->sv_indices + m_model->l, &model->sv_indices);
        std::copy_n(m_model->sv_indices, m_model->l, model->sv_indices);
    	
        model->SV = new svm_node*[m_model->l];
    	for(auto i = 0; i < m_model->l; i++)
    	{
            model->SV[i] = m_problem.x[m_model->sv_indices[i]];
           /* model->SV[i] = new svm_node[dims + 1];
            int filledIndex = 0;
            for (unsigned j = 0; j < dims; j++)
            {
                if (model->SV[i][j].value != 0)
                {
                    model->SV[i][filledIndex].index = m_model->SV[i][j].index;
                    model->SV[i][filledIndex].value = m_model->SV[i][j].value;
                    filledIndex++;
                }
            }
            model->SV[i][filledIndex].index = -1;
            model->SV[i][filledIndex].value = 0;*/
    	}

       

    	//probA, probB not used and not copied
        model->probA = model->probB = nullptr;

        return model;
    }
	
public:
    std::shared_ptr<ISvm> clone() override
    {
    	//make sure this is not memory leak
    	//svm_parameter, svm_model, svm_problem does not have proper deep copy mechanism 
    	//auto ptr = std::shared_ptr<libSvmImplementation>(new libSvmImplementation(*this), [=](phd::svm::libSvmImplementation*) {});
        //auto ptr = std::make_shared<libSvmImplementation>(*this);

        this->save("temp.xml");
    	auto ptr = std::make_shared<libSvmImplementation>("temp.xml");

       
        //ptr->m_featureSet = m_featureSet;
        //ptr->m_numberOfFeatures = m_numberOfFeatures;
        //ptr->m_gammas = m_gammas;
        //ptr->m_certaintyNegative = m_certaintyNegative;
        //ptr->m_certaintyPositive = m_certaintyPositive;
        //ptr->m_param = m_param;
        //ptr->m_param.weight = nullptr;
        //ptr->m_param.weight_label = nullptr;
        //ptr->m_sv = m_sv;
        //ptr->m_optimalThresholdSet = m_optimalThresholdSet;
        //ptr->m_optimalProbabilityThreshold = m_optimalProbabilityThreshold;
    	
        //ptr->m_param.gammas = m_param.gammas;
        //ptr->m_param.gammas_after_training = m_param.gammas_after_training;
        //ptr->m_problem = copy();
        //ptr->m_model = copy_model();
    	
        return ptr;
    }

	private:
	std::vector<svmComponents::Feature> m_featureSet;
	int m_numberOfFeatures;
    std::vector<double> m_gammas;

    //std::shared_ptr<svmComponents::IGroupPropagation> m_groupStrategy;
    std::string m_groupStrategyName = "None";

};

inline double libSvmImplementation::getC() const
{
    return m_param.C;
}

inline double libSvmImplementation::getGamma() const
{
    return m_param.gamma;
}

inline double libSvmImplementation::getCoef0() const
{
    return m_param.coef0;
}

inline double libSvmImplementation::getDegree() const
{
    return m_param.degree;
}

inline double libSvmImplementation::getNu() const
{
    return m_param.nu;
}

inline double libSvmImplementation::getP() const
{
    return m_param.p;
}

inline void libSvmImplementation::setC(double value)
{
    if (value > 0.0)
    {
        m_param.C = value;
    }
    else
    {
        throw ValueNotPositiveException("C");
    }
}

inline void libSvmImplementation::setCoef0(double value)
{
    if (value > 0.0)
    {
        m_param.coef0 = value;
    }
    else
    {
        throw ValueNotPositiveException("Coef0");
    }
}

inline void libSvmImplementation::setDegree(double value)
{
    if (value > 0.0)
    {
        m_param.degree = static_cast<int>(value);
    }
    else
    {
        throw ValueNotPositiveException("degree");
    }
}

inline void libSvmImplementation::setGamma(double value)
{
    if (value > 0.0)
    {
        m_param.gamma = value;
    }
    else
    {
        throw ValueNotPositiveException("gamma");
    }
}

inline void libSvmImplementation::setGammas(const std::vector<double>& value)
{
    m_gammas = value;
    m_param.gammas = &m_gammas;
   /* }
    else
    {
        throw ValueNotPositiveException("gamma");
    }*/
}

inline void libSvmImplementation::setNu(double value)
{
        const auto minValue = 0.0;
        const auto maxValue = 1.0;
        if (value > minValue && value < maxValue)
        {
            m_param.nu = value;
        }
        else
        {
            throw ValueNotInRange("Nu", value, minValue, maxValue);
        }
}

inline void libSvmImplementation::setP(double value)
{
    if (value > 0.0)
    {
        m_param.p = value;
    }
    else
    {
        throw ValueNotPositiveException("p (svr epsilon)");
    }
}
}} // namespace phd::svm
