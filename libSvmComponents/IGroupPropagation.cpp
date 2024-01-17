#include <libSvmComponents/IGroupPropagation.h>

namespace svmComponents
{
std::unordered_map<std::string, GroupStrategy> GroupsStrategyFactory::m_translations =
{
	{"Median", GroupStrategy::MedianAnswer},
	{"Max", GroupStrategy::MaxAnswer},
	{"Average", GroupStrategy::AverageAnswer},
	{"None", GroupStrategy::None},
};
const std::unordered_map<GroupStrategy, std::string> GroupsStrategyFactory::m_translationsToStr =
{
	{GroupStrategy::MedianAnswer, "Median"},
	{GroupStrategy::MaxAnswer, "Max"},
	{GroupStrategy::AverageAnswer, "Average"},
	{GroupStrategy::None, "None"},
};
} // namespace svmComponents
