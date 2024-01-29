#pragma once

#include <algorithm>
#include <string>

#include <vector>
//#include "libFileSystem/FileSystem.h"
#include "SvmLib/libSvmImplementation.h"

std::vector<std::filesystem::path> getAllFeaturesSets(std::filesystem::path folderPath);

std::vector<std::filesystem::path> getAllSvms(std::filesystem::path folderPath);

uint64_t classifyMulticlass(int classes,
                            std::vector<std::shared_ptr<phd::svm::libSvmImplementation>>& classifiers,
                            std::vector<std::map<int, int>>& labelsMapping,
                            const gsl::span<const float>& sample);

void getLabelMapping(std::vector<std::string>::size_type numberOfClasses, std::vector<std::map<int, int>>& labelsMapping);

void join_multiclass();

void createAnswerTargegFiles();


void rerun_models();

//Creates ensembleLog__.txt for each model which is used for some visualizations in Python
void rerun_regions_for_plots_of_nodes();

//Reruns EnsembleList (single list) when no regions were saved
void rerun_regions();

double get_margin(std::shared_ptr<phd::svm::ISvm> svm, const dataset::Dataset<std::vector<float>, float>& validation_set);

//double get_margin_sv(std::shared_ptr<phd::svm::ISvm> svm);

void rerun_models2(std::string basePath_, std::string outputPath, std::string variantName, std::string test_val, std::string metric);
