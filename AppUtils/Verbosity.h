#pragma once

#include <string>
#include "libPlatform/EnumStringConversions.h"

namespace testApp
{
enum class Verbosity
{
	None,
	Minimal,
	Standard,
	All,
};

inline std::unordered_map<std::string, Verbosity> getVerbosityTranslation()
{
	const std::unordered_map<std::string, Verbosity> verbosityTranslations =
	{
		{"None", Verbosity::None},
		{"Minimal", Verbosity::Minimal},
		{"Standard", Verbosity::Standard},
		{"All", Verbosity::All},
	};
	return verbosityTranslations;
}

inline Verbosity verbosityFromString(std::string v)
{
	return platform::stringToEnum(v, getVerbosityTranslation());
}

inline std::istream& operator>>(std::istream& in, testApp::Verbosity& verbosity)
{
	std::string token;
	in >> token;
	if (token == "All")
		verbosity = Verbosity::All;
	else if (token == "Standard")
		verbosity = Verbosity::Standard;
	else if (token == "Minimal")
		verbosity = Verbosity::Minimal;
	else if (token == "None")
		verbosity = Verbosity::None;
	else
		verbosity = Verbosity::Standard;
	return in;
}

inline std::ostream& operator<<(std::ostream& out, const testApp::Verbosity& /*verbosity*/)
{
	out << "Not implemented ostream for verbosity\n";
	return out;
}
} // namespace testApp
