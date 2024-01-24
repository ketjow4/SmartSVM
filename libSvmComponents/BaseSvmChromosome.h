
#pragma once
#include <chrono>
#include <optional>
#include "libGeneticComponents/BaseChromosome.h"
#include "ConfusionMatrix.h"
#include "SvmLib/ISvm.h"
#include "Metric.h"

namespace svmComponents
{
class BaseSvmChromosome : public virtual geneticComponents::BaseChromosome
{
public:
    BaseSvmChromosome() = default;

    unsigned long getNumberOfSupportVectors() const;

    virtual void updateClassifier(std::shared_ptr<phd::svm::ISvm> classifier);
    std::shared_ptr<phd::svm::ISvm> getClassifier() const;

    void updateTime(std::chrono::duration<double, std::milli> classificationTime);
    std::chrono::duration<double, std::milli> getTime() const;

    void updateConfusionMatrix(const std::optional<ConfusionMatrix>& matrix);
    std::optional<ConfusionMatrix> getConfusionMatrix() const;


    void updateMetric(Metric metric)
    {
        m_metric = metric;
    }
    Metric getMetric() const
    {
        return m_metric;
    }

  //  BaseSvmChromosome(const BaseSvmChromosome& rhs)
	 //   : m_classifier([&]()
  //  {
  //              if (rhs.m_classifier)
  //                  return rhs.m_classifier->clone();
  //              return std::shared_ptr<phd::svm::ISvm>(nullptr);
  //  			
  //  }())
	 //   , m_classificationTime(rhs.m_classificationTime)
		//, m_confusionMatrix(rhs.m_confusionMatrix)

  //  {
	 //   // Handled by initializer list
  //  }

  //  BaseSvmChromosome(BaseSvmChromosome&& rhs) : m_classifier(std::move(rhs.m_classifier))
  //                                             , m_classificationTime(std::move(rhs.m_classificationTime))
		//									   , m_confusionMatrix(std::move(rhs.m_confusionMatrix))
  //  {
	 //   // Handled by initializer list
  //  }
	
  //  BaseSvmChromosome& operator= (BaseSvmChromosome rhs) {
  //      std::swap(m_classificationTime, rhs.m_classificationTime);
  //      //m_classifier = rhs.m_classifier;
  //      std::swap(m_classifier, rhs.m_classifier);
  //      std::swap(m_confusionMatrix, rhs.m_confusionMatrix);
  //      std::swap(m_fitness, rhs.m_fitness);
  //      return *this;
  //  }
	
protected:
    std::shared_ptr<phd::svm::ISvm> m_classifier;
    std::chrono::duration<double, std::milli> m_classificationTime;
    std::optional<ConfusionMatrix> m_confusionMatrix;
    Metric m_metric;
};


inline unsigned long BaseSvmChromosome::getNumberOfSupportVectors() const
{
    return m_classifier->getNumberOfSupportVectors();
}

inline void BaseSvmChromosome::updateClassifier(std::shared_ptr<phd::svm::ISvm> classifier)
{
    m_classifier = std::move(classifier);
}

inline std::shared_ptr<phd::svm::ISvm> BaseSvmChromosome::getClassifier() const
{
    return m_classifier;
}

inline void BaseSvmChromosome::updateTime(std::chrono::duration<double, std::milli> classificationTime)
{
    m_classificationTime = classificationTime;
}

inline std::chrono::duration<double, std::milli> BaseSvmChromosome::getTime() const
{
    return m_classificationTime;
}

inline void BaseSvmChromosome::updateConfusionMatrix(const std::optional<ConfusionMatrix>& matrix)
{
    if (matrix)
    {
        m_confusionMatrix.emplace(std::move(matrix.value()));
    }
}

inline std::optional<ConfusionMatrix> BaseSvmChromosome::getConfusionMatrix() const
{
    return m_confusionMatrix;
}
} // namespace svmComponents
