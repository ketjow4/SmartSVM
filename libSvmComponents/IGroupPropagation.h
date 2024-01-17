#pragma once


#include <vector>
#include <algorithm>
#include <numeric>
#include <memory>

#include "libPlatform/EnumStringConversions.h"

namespace svmComponents
{
enum class GroupStrategy
{
	MaxAnswer,
	AverageAnswer,
	MedianAnswer,
	None
};

class IGroupPropagation
{
public:
	

	virtual ~IGroupPropagation() = default;

	virtual float propagateAnswer(const std::vector<float>& answers) = 0;
	virtual GroupStrategy getType() = 0;
};


class MaxAnswerPropagation : public IGroupPropagation
{
public:
	float propagateAnswer(const std::vector<float>& answers) override
	{
		return *std::max_element(answers.begin(), answers.end());
	}

	GroupStrategy getType() override
	{
		return  GroupStrategy::MaxAnswer;
	}
};

class AverageAnswerPropagation : public IGroupPropagation
{
public:
	float propagateAnswer(const std::vector<float>& answers) override
	{
		float sum = 0.0;
		return std::accumulate(answers.begin(), answers.end(), sum) / static_cast<float>(answers.size());
	}

	GroupStrategy getType() override
	{
		return  GroupStrategy::AverageAnswer;
	}
};

class MedianAnswerPropagation : public IGroupPropagation
{
public:
	float propagateAnswer(const std::vector<float>& answers) override
	{
		//https://en.cppreference.com/w/cpp/algorithm/nth_element
		std::vector<float> copy = answers;
		size_t n = answers.size() / 2;
		nth_element(copy.begin(), copy.begin() + n, copy.end());
		return copy[n];
	}

	GroupStrategy getType() override
	{
		return  GroupStrategy::MedianAnswer;
	}
};

class GroupsStrategyFactory 
{
public:
	static std::string convert(GroupStrategy gs)
	{
		return platform::enumToString(gs, m_translationsToStr);
	}

	static GroupStrategy convert(const std::string& name)
	{
		return platform::stringToEnum(name, m_translations);
	}

	static std::shared_ptr<IGroupPropagation> create(GroupStrategy gs)
	{
		switch (gs)
		{
		case GroupStrategy::MaxAnswer:
			return std::make_shared<svmComponents::MaxAnswerPropagation>();
		case GroupStrategy::AverageAnswer:
			return std::make_shared<svmComponents::AverageAnswerPropagation>();
		case GroupStrategy::MedianAnswer:
			return std::make_shared<svmComponents::MedianAnswerPropagation>();
		case GroupStrategy::None:
			return nullptr;
		default:
			return nullptr;
		}
	}


private:
	static std::unordered_map<std::string, GroupStrategy> m_translations;
	const static std::unordered_map<GroupStrategy, std::string> m_translationsToStr;

};


} // namespace svmComponents
