
#include "SvmValidationFactory.h"


namespace svmComponents
{
const  std::unordered_map<std::string, SvmValidation> SvmValidationFactory::m_translations =
{
	{"Regular", SvmValidation::Regular},
	{"Subset", SvmValidation::Subset}
};

} // namespace svmComponents