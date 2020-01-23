#include <memory>

#include <iostream>
#include <algorithm>
#include <vector>

#include "frontend_agglomerate.h"
#include "evaluate.hpp"
#include "backend/MergeFunctions.hpp"
#include "backend/basic_watershed.hpp"
#include "backend/region_graph.hpp"

std::map<int, WaterzContext*> WaterzContext::_contexts;
int WaterzContext::_nextId = 0;

WaterzState
initialize(
		std::size_t     width,
		std::size_t     height,
		std::size_t     depth,
		const AffValue* affinity_data,
		SegID*          segmentation_data,
		const GtID*     ground_truth_data,
		AffValue        affThresholdLow,
		AffValue        affThresholdHigh,
		bool            findFragments,
		UnmergeGroupListTupleList* unmergeList) {

	std::size_t num_voxels = width*height*depth;

	// wrap affinities (no copy)
	affinity_graph_ref<AffValue> affinities(
			affinity_data,
			boost::extents[3][width][height][depth]
	);

	// wrap segmentation array (no copy)
	volume_ref_ptr<SegID> segmentation(
			new volume_ref<SegID>(
					segmentation_data,
					boost::extents[width][height][depth]
			)
	);

	counts_t<std::size_t> sizes;

	if (findFragments) {

		watershed(affinities, affThresholdLow, affThresholdHigh, *segmentation, sizes);

	} else {

		std::size_t maxId = *std::max_element(segmentation_data, segmentation_data + num_voxels);
		sizes.resize(maxId + 1);
		for (std::size_t i = 0; i < num_voxels; i++)
			sizes[segmentation_data[i]]++;
	}

	std::size_t numNodes = sizes.size();

	std::shared_ptr<RegionGraphType> regionGraph(
			new RegionGraphType(numNodes)
	);

	std::shared_ptr<StatisticsProviderType> statisticsProvider(
			new StatisticsProviderType(*regionGraph)
	);

	get_region_graph(
			affinities,
			*segmentation,
			numNodes - 1,
			*statisticsProvider,
			*regionGraph);

	std::shared_ptr<ScoringFunctionType> scoringFunction(
			new ScoringFunctionType(*regionGraph, *statisticsProvider)
	);

	std::shared_ptr<RegionMergingType> regionMerging(
			new RegionMergingType(*regionGraph)
	);

	WaterzContext* context = WaterzContext::createNew();
	context->regionGraph        = regionGraph;
	context->regionMerging      = regionMerging;
	context->scoringFunction    = scoringFunction;
	context->statisticsProvider = statisticsProvider;
	context->segmentation       = segmentation;

	WaterzState initial_state;
	initial_state.context = context->id;

	if (ground_truth_data != NULL) {

		// wrap ground-truth (no copy)
		volume_const_ref_ptr<GtID> groundtruth(
				new volume_const_ref<GtID>(
						ground_truth_data,
						boost::extents[width][height][depth]
				)
		);

		context->groundtruth = groundtruth;
	}

	context->unmergeList = *unmergeList;

	return initial_state;
}

std::vector<Merge>
mergeUntil(
		WaterzState& state,
		float        threshold) {

	WaterzContext* context = WaterzContext::get(state.context);

	std::cout << "merging until threshold " << threshold << std::endl;

	std::vector<Merge>  mergeHistory;
	MergeHistoryVisitor mergeHistoryVisitor(mergeHistory);

	if (context->unmergeList.size()) {
		UnmergeTracker* unmergeTracker = new UnmergeTracker(context->unmergeList);
		mergeHistoryVisitor.setUnmergeTracker(unmergeTracker);
	}

	std::size_t merged = context->regionMerging->mergeUntil(
			*context->scoringFunction,
			*context->statisticsProvider,
			threshold,
			mergeHistoryVisitor);

	if (merged) {

		std::cout << "extracting segmentation" << std::endl;

		context->regionMerging->extractSegmentation(*context->segmentation);
	}

	if (context->groundtruth) {

		std::cout << "evaluating current segmentation against ground-truth" << std::endl;

		auto m = compare_volumes(*context->groundtruth, *context->segmentation);

		state.metrics.rand_split = std::get<0>(m);
		state.metrics.rand_merge = std::get<1>(m);
		state.metrics.voi_split  = std::get<2>(m);
		state.metrics.voi_merge  = std::get<3>(m);
	}

	// TODO: make unmergeTracker a shared pointer
	// delete unmergeTracker;
	return mergeHistory;
}

std::vector<ScoredEdge>
getRegionGraph(WaterzState& state) {

	WaterzContext* context = WaterzContext::get(state.context);
	std::shared_ptr<RegionMergingType> regionMerging = context->regionMerging;
	std::shared_ptr<ScoringFunctionType> scoringFunction = context->scoringFunction;

	return regionMerging->extractRegionGraph<ScoredEdge>(*scoringFunction);
}

void
free(WaterzState& state) {

	WaterzContext::free(state.context);
}
