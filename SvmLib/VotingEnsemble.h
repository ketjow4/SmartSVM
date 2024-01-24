#pragma once

#include <filesystem>
#include <fstream>
#include "libPlatform/loguru.hpp"
#include "ISvm.h"
#include "SvmExceptions.h"
#include "libSvmInternal.h"
#include "libSvmImplementation.h"
#include "ExtraTreeWrapper.h"
#include "EnsembleListSvm.h"

namespace phd {
	namespace svm
	{
		class VotingEnsemble : public phd::svm::ISvm
		{
		public:
			VotingEnsemble(std::vector<std::shared_ptr<phd::svm::EnsembleListSvm>> classifieres, std::vector<double> weights)
				: m_weights(std::move(weights))
				  , m_classifieres(std::move(classifieres)),
				fifty_percent_certain_(false)
			{
				if (m_classifieres.size() != m_weights.size())
				{
					throw std::exception("wrong size of weights or classifiers");
				}
			}


			VotingEnsemble(std::vector<std::filesystem::path> classifiers, std::filesystem::path weights, std::string classification_type = "", bool fiftyPercentCertain = false):
				classification_type_(std::move(classification_type)),
				fifty_percent_certain_(fiftyPercentCertain)
			{
				for (auto i = 0u; i < classifiers.size(); ++i)
				{
					m_classifieres.emplace_back(std::make_shared<phd::svm::EnsembleListSvm>(classifiers[i], true));
				}


				if (std::filesystem::exists(weights))
				{
					std::ifstream weightsFile(weights);
					double weight;
					while(weightsFile >> weight)
					{
						m_weights.emplace_back(weight);
					}
				}
				else
				{
					m_weights = std::vector<double>(m_classifieres.size(), 1);	
				}
			}


			void scoreLevelWise(const dataset::Dataset<std::vector<float>, float>& dataset)
			{
				for(auto i = 0u; i < m_classifieres.size(); ++i)
				{
					std::vector<std::array<std::array<uint32_t, 2>, 2>> cm(m_classifieres[i]->list_length);
					auto samples = dataset.getSamples();
					auto labels = dataset.getLabels();

					#pragma omp parallel for
					for(int j = 0; j < static_cast<int>(samples.size()); ++j)
					{
						auto [respond, nodeId] = m_classifieres[i]->LastNodeSchemeAndNode(samples[j]);
						if(respond != -100)
						{
							#pragma omp critical
							++cm[nodeId][static_cast<int>(respond)][static_cast<int>(labels[j])];
						}
					}

					std::vector<double> weights(m_classifieres[i]->list_length);

					for(auto j = 0; j < m_classifieres[i]->list_length; ++j)
					{
						weights[j] = svmComponents::ConfusionMatrix(cm[j]).MCC();
					}		
					m_classifieres[i]->m_levelWeights = weights;
				}
			}



			float classifyNodeWeightsAndLog(const gsl::span<const float> sample) const
			{
				std::ofstream analyze("D:\\ENSEMBLE_910_scalling_thr_001_k32_RBF\\output_test.txt", std::ios_base::app);

				std::vector<float> answers;
				std::vector<double> nodesWeights;

				auto final_answer = 0.0;
				for (auto i = 0u; i < m_classifieres.size(); i++)
				{
					auto [response, nodeId] = m_classifieres[i]->LastNodeSchemeAndNode(sample);

					answers.emplace_back(response);
					nodesWeights.emplace_back(m_classifieres[i]->m_levelWeights[nodeId]);

					if (response == 0)
					{
						final_answer += -1.0 * m_classifieres[i]->m_levelWeights[nodeId];
					}
					else if (response == 1)
					{
						final_answer += 1.0 * m_classifieres[i]->m_levelWeights[nodeId];
					}
					//if response = -100 do nothing
				}

				auto count = 0;
				for (auto i = 0u; i < answers.size(); ++i)
				{
					analyze << answers[i];
					if (answers[i] == -100)
					{
						count++;
						analyze  << ", ";
					}
					else
					{
						analyze << " * " << nodesWeights[i] << ", ";
					}
				}
				analyze << "  Unsure %: " << static_cast<float>(count) / static_cast<float>(answers.size()) << "  Final answer: " << std::to_string(final_answer);
				analyze << "\n";

				if (final_answer == 0.0)
				{
					/*			for (auto i = 0u; i < m_classifieres.size(); i++)
								{
									auto [response, nodeId] = m_classifieres[i]->NewScheme(sample);

									if (response == 0)
									{
										final_answer += -1.0 * m_classifieres[i]->m_levelWeights[nodeId];
									}
									else if (response == 1)
									{
										final_answer += 1.0 * m_classifieres[i]->m_levelWeights[nodeId];
									}
								}
								return final_answer > 0 ? 1.0f : 0.0f;*/

					analyze << "Got zero, regular classification";
					analyze << "\n";

					return classify(sample);
					//return 0;
					//LOG_F(ERROR, "uncertain examples");
					//return -100; //temporary solution
				}

				return final_answer > 0 ? 1.0f : 0.0f;
			}

#pragma optimize("", off)

			
			float classifyNodeWeights(const gsl::span<const float> sample) const
			{
				auto final_answer = 0.0;
				for (auto i = 0u; i < m_classifieres.size(); i++)
				{
					auto [response, nodeId] = m_classifieres[i]->LastNodeSchemeAndNode(sample);
					
					if (response == 0)
					{
						final_answer += -1.0 * m_classifieres[i]->m_levelWeights[nodeId];
					}
					else if (response == 1)
					{
						final_answer += 1.0 * m_classifieres[i]->m_levelWeights[nodeId];
					}
					//if response = -100 do nothing
				}

				if (final_answer == 0.0)
				{
		/*			for (auto i = 0u; i < m_classifieres.size(); i++)
					{
						auto [response, nodeId] = m_classifieres[i]->NewScheme(sample);

						if (response == 0)
						{
							final_answer += -1.0 * m_classifieres[i]->m_levelWeights[nodeId];
						}
						else if (response == 1)
						{
							final_answer += 1.0 * m_classifieres[i]->m_levelWeights[nodeId];
						}
					}
					return final_answer > 0 ? 1.0f : 0.0f;*/

					return classify(sample);
					//return 0;
					//LOG_F(ERROR, "uncertain examples");
					//return -100; //temporary solution
				}

				return final_answer > 0 ? 1.0f : 0.0f;
			}

#pragma optimize("", on)
			
			void save(const std::filesystem::path& filepath) override
			{
				std::ofstream weigthsFile(filepath.string() + "__weights.txt");
				
				for(auto i = 0u; i < m_classifieres.size(); ++i)
				{
					m_classifieres[i]->save(filepath.string() + "__" + std::to_string(i) + ".xml");
					weigthsFile.precision(18);
					weigthsFile << m_weights[i] << "\n";
				}
				weigthsFile.close();
			}

			bool isTrained() const override { return true; }
			bool canGiveProbabilityOutput() const override { return false; }
			bool canClassifyWithOptimalThreshold() const override { return true; }
			
			uint32_t getNumberOfSupportVectors() const override
			{
				/*return	std::accumulate(m_classifieres.begin(), m_classifieres.end(), uint32_t(0),
				                        [](auto a, auto b)
				                        {
					                        return a->getNumberOfSupportVectors() + b->getNumberOfSupportVectors();
				                        });			*/
				return 0;
			}
			std::vector<std::vector<float>> getSupportVectors() const override
			{
				LOG_F(WARNING, "getSupportVectors voting ensemble not implemented");
				return std::vector<std::vector<float>>();
			}

			
			float classify(const gsl::span<const float> sample) const override
			{
				auto final_answer = 0.0;
				for(auto i = 0u; i < m_classifieres.size(); i++)
				{
					auto response = m_classifieres[i]->classifyWithCertainty(sample); 

					if(response == 0)
					{
						final_answer += -1.0 * m_weights[i];
					}
					else if (response == 1)
					{
						final_answer += 1.0 * m_weights[i];
					}
					//if response = -100 do nothing
				}

				if (final_answer == 0.0)
				{
					return 0;
					//return -100; //temporary solution
				}

				return final_answer > 0 ? 1.0f : 0.0f;
			}


			float classify_no_weights(const gsl::span<const float> sample) const
			{
				auto final_answer = 0.0;
				for (auto i = 0u; i < m_classifieres.size(); i++)
				{
					auto response = m_classifieres[i]->classifyWithCertainty(sample);

					if (response == 0)
					{
						final_answer += -1.0;
					}
					else if (response == 1)
					{
						final_answer += 1.0;
					}
					//if response = -100 do nothing
				}

				if (final_answer == 0.0)
				{
					return 0;
					//return -100; //temporary solution
				}

				return final_answer > 0 ? 1.0f : 0.0f;
			}


			float classify_all(const gsl::span<const float> sample) const
			{
				auto final_answer = 0.0;
				for (auto i = 0u; i < m_classifieres.size(); i++)
				{
					auto response = m_classifieres[i]->classifyAll(sample);

					if (response >= 0)
					{
						final_answer += -1.0;
					}
					else if (response < 0)
					{
						final_answer += 1.0;
					}
				}

				return final_answer > 0 ? 1.0f : 0.0f;
			}


			float classification_tests(const gsl::span<const float> sample) const
			{
				if(classification_type_ == "All")
				{
					return classify_all(sample);
				}
				else if (classification_type_ == "Cascade")
				{
					return classify_no_weights(sample);
				}
				else if (classification_type_ == "Cascade_weight")
				{
					return classify(sample);
				}
				else if (classification_type_ == "Cascade_node")
				{
					return classifyNodeWeights(sample);
				}
				throw std::exception("Wrong classification_type_ in classification_tests method of Voting Ensemble");
			}

			//BEFORE 22.07.2023 this was the calculated
			float classifyHyperplaneDistance(const gsl::span<const float> sample) const override { return classifyNodeWeights(sample); }
			double classificationProbability(const gsl::span<const float> sample) const override { return classifyNodeWeights(sample); }
			double classifyWithOptimalThreshold(const gsl::span<const float> sample) const override { return classifyNodeWeights(sample); }

		/*	float classifyHyperplaneDistance(const gsl::span<const float> sample) const override { return classification_tests(sample); }
			double classificationProbability(const gsl::span<const float> sample) const override { return classification_tests(sample); }
			double classifyWithOptimalThreshold(const gsl::span<const float> sample) const override { return classification_tests(sample); }*/

			float classifyWithCertainty(const gsl::span<const float> sample) const override
			{
				//if (classification_type_ == "All")
				//{
				//	return 1.0; //this only indicate that there is always some answer - for classification experiment only
				//}
				//else if (classification_type_ == "Cascade")
				//{
				//	auto final_answer = 0.0;
				//	auto countUncertain = 0;
				//	//for(auto& cl : m_classifieres)
				//	for (auto i = 0u; i < m_classifieres.size(); i++)
				//	{
				//		auto response = m_classifieres[i]->classifyWithCertainty(sample);
				//		if (response == 0)
				//		{
				//			final_answer += -1.0;
				//		}
				//		else if (response == 1)
				//		{
				//			final_answer += 1.0;
				//		}
				//		//if response = -100 do nothing
				//		else if (response == -100)
				//		{
				//			countUncertain++;
				//		}
				//	}

				//	if (fifty_percent_certain_ && countUncertain >= m_classifieres.size() / 2)
				//	{
				//		return -100;
				//	}

				//	if (final_answer == 0.0)
				//	{
				//		return -100;
				//	}

				//	return final_answer > 0 ? 1.0f : 0.0f;
				//}
				//else if (classification_type_ == "Cascade_weight")
				//{
				//	auto final_answer = 0.0;
				//	auto countUncertain = 0;
				//	//for(auto& cl : m_classifieres)
				//	for (auto i = 0u; i < m_classifieres.size(); i++)
				//	{
				//		auto response = m_classifieres[i]->classifyWithCertainty(sample);
				//		if (response == 0)
				//		{
				//			final_answer += -1.0 * m_weights[i];
				//		}
				//		else if (response == 1)
				//		{
				//			final_answer += 1.0 * m_weights[i];
				//		}
				//		//if response = -100 do nothing
				//		else if (response == -100)
				//		{
				//			countUncertain++;
				//		}
				//	}

				//	if (fifty_percent_certain_ && countUncertain >= m_classifieres.size() / 2)
				//	{
				//		return -100;
				//	}

				//	if (final_answer == 0.0)
				//	{
				//		return -100;
				//	}

				//	return final_answer > 0 ? 1.0f : 0.0f;
				//}
				//else if (classification_type_ == "Cascade_node")
				//{
				//	auto final_answer = 0.0;
				//	auto countUncertain = 0;
				//	for (auto i = 0u; i < m_classifieres.size(); i++)
				//	{
				//		auto [response, nodeId] = m_classifieres[i]->LastNodeSchemeAndNode(sample);

				//		if (response == 0)
				//		{
				//			final_answer += -1.0 * m_classifieres[i]->m_levelWeights[nodeId];
				//		}
				//		else if (response == 1)
				//		{
				//			final_answer += 1.0 * m_classifieres[i]->m_levelWeights[nodeId];
				//		}
				//		else if (response == -100)
				//		{
				//			countUncertain++;
				//		}
				//	}

				//	if (fifty_percent_certain_ && countUncertain >= m_classifieres.size() / 2)
				//	{
				//		return -100;
				//	}

				//	if (final_answer == 0.0)
				//	{
				//		return -100;
				//	}

				//	return final_answer > 0 ? 1.0f : 0.0f;
				//}



				auto final_answer = 0.0;
				auto countUncertain = 0;
				//for(auto& cl : m_classifieres)
				for (auto i = 0u; i < m_classifieres.size(); i++)
				{
					auto response = m_classifieres[i]->classifyWithCertainty(sample);
					if (response == 0)
					{
						final_answer += -1.0 * m_weights[i];
					}
					else if (response == 1)
					{
						final_answer += 1.0 * m_weights[i];
					}
					//if response = -100 do nothing
					else if (response == -100)
					{
						countUncertain++;
					}
				}

				if(countUncertain >= m_classifieres.size() / 2  )
				{
					return -100;
				}

				if (final_answer == 0.0)
				{
					return -100;
				}

				return final_answer > 0 ? 1.0f : 0.0f;
			}


			std::unordered_map<int, int> classifyGroups(const dataset::Dataset<std::vector<float>, float>& /*dataWithGroups*/) const
			{
				throw std::exception("Classyfing groups not implemented in VotingEnsemble");
			}

			std::unordered_map<int, float> classifyGroupsRawScores(const dataset::Dataset<std::vector<float>, float>& /*dataWithGroups*/) const override
			{
				throw std::exception("classifyGroupsRawScores not implemented in VotingEnsemble");
			}

			std::shared_ptr<ISvm> clone() override { throw std::exception("Not implemented Voting Ensemble 1"); }
			double getC() const override { throw std::exception("Not implemented Voting Ensemble 2"); }
			double getGamma() const override { throw std::exception("Not implemented Voting Ensemble 3"); }
			std::vector<double> getGammas() const override { throw std::exception("Not implemented Voting Ensemble 4"); }
			double getCoef0() const override { throw std::exception("Not implemented Voting Ensemble 5"); }
			double getDegree() const override { throw std::exception("Not implemented Voting Ensemble 6"); }
			double getNu() const override { throw std::exception("Not implemented Voting Ensemble 7"); }
			double getP() const override { throw std::exception("Not implemented Voting Ensemble 8"); }
			double getT() const override { throw std::exception("Not implemented Voting Ensemble 9"); }
			void setC(double /*value*/) override { throw std::exception("Not implemented Voting Ensemble 10"); }
			void setCoef0(double /*value*/) override { throw std::exception("Not implemented Voting Ensemble 11"); }
			void setDegree(double /*value*/) override { throw std::exception("Not implemented Voting Ensemble 12"); }
			void setGamma(double /*value*/) override { throw std::exception("Not implemented Voting Ensemble 13"); }
			void setGammas(const std::vector<double>& /*value*/) override { throw std::exception("Not implemented Voting Ensemble 14"); }
			void setNu(double /*value*/) override { throw std::exception("Not implemented Voting Ensemble 15"); }
			void setP(double /*value*/) override { throw std::exception("Not implemented Voting Ensemble 16"); }
			void setT(double /*value*/) override { throw std::exception("Not implemented Voting Ensemble 17"); }
			void setOptimalProbabilityThreshold(double /*optimalThreshold*/) override { return; }
			KernelTypes getKernelType() const override { return phd::svm::KernelTypes::Custom; }
			void setKernel(KernelTypes /*kernelType*/) override { throw std::exception("Not implemented Voting Ensemble 19"); }
			SvmTypes getType() const override { throw std::exception("Not implemented Voting Ensemble 20"); }
			void setType(SvmTypes /*svmType*/) override { throw std::exception("Not implemented Voting Ensemble 21"); }
			void train(const dataset::Dataset<std::vector<float>, float>& /*trainingSet*/, bool /*probabilityNeeded*/) override { throw std::exception("Not implemented Voting Ensemble 22"); }
			//void setTerminationCriteria(const cv::TermCriteria& /*value*/) override { throw std::exception("Not implemented Voting Ensemble 23"); }
			//cv::TermCriteria getTerminationCriteria() const override { throw std::exception("Not implemented Voting Ensemble 24"); }
			void setFeatureSet(const std::vector<svmComponents::Feature>& /*features*/, int /*numberOfFeatures*/) override { throw std::exception("Not implemented Voting Ensemble 25"); }
			const std::vector<svmComponents::Feature>& getFeatureSet() override { throw std::exception("Not implemented Voting Ensemble 26"); }
			uint32_t getNumberOfKernelParameters(KernelTypes /*kernelType*/) const override { throw std::exception("Not implemented Voting Ensemble 27"); }

			std::vector<double> m_weights;
			std::vector<std::shared_ptr<phd::svm::EnsembleListSvm>> m_classifieres;

			std::shared_ptr<phd::svm::libSvmImplementation> m_final_classifier;
			std::string classification_type_;
			bool fifty_percent_certain_;

		private:
			/*std::vector<double> m_weights;
			std::vector<std::shared_ptr<phd::svm::EnsembleListSvm>> m_classifieres;*/
		};
	}
} // namespace phd::svm
