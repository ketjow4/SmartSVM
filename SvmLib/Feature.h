#pragma once

namespace svmComponents
{
struct Feature
{
	std::uint64_t id;

	Feature() : id(0)
	{
	}

	Feature(std::uint64_t id) : id(id)
	{
	}

	bool operator==(const Feature& other) const
	{
		return id == other.id;
	}

	bool operator<(const Feature& other) const
	{
		return id < other.id;
	}
};
} // namespace svmComponents
