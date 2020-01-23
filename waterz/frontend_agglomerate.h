#ifndef C_FRONTEND_H
#define C_FRONTEND_H

#include <vector>
#include <unordered_map>

#include "backend/IterativeRegionMerging.hpp"
#include "backend/MergeFunctions.hpp"
#include "backend/Operators.hpp"
#include "backend/types.hpp"
#include "backend/BinQueue.hpp"
#include "backend/PriorityQueue.hpp"
#include "backend/HistogramQuantileProvider.hpp"
#include "backend/VectorQuantileProvider.hpp"
#include "evaluate.hpp"

typedef uint64_t SegID;
typedef uint32_t GtID;
typedef float AffValue;
typedef float ScoreValue;
typedef RegionGraph<SegID> RegionGraphType;

// to be created by __init__.py
#include <ScoringFunction.h>
#include <Queue.h>

typedef typename ScoringFunctionType::StatisticsProviderType StatisticsProviderType;
typedef IterativeRegionMerging<SegID, ScoreValue, QueueType> RegionMergingType;
typedef std::vector<std::vector<std::vector<uint64_t>>> UnmergeGroupListTupleList;

struct Metrics {

	double voi_split;
	double voi_merge;
	double rand_split;
	double rand_merge;
};

struct Merge {

	SegID a;
	SegID b;
	SegID c;
	ScoreValue score;
};

struct ScoredEdge {

	ScoredEdge(SegID u_, SegID v_, ScoreValue score_) :
		u(u_),
		v(v_),
		score(score_) {}

	SegID u;
	SegID v;
	ScoreValue score;
};

struct WaterzState {

	int     context;
	Metrics metrics;
};

class WaterzContext {

public:

	static WaterzContext* createNew() {

		WaterzContext* context = new WaterzContext();
		context->id = _nextId;
		_nextId++;
		_contexts.insert(std::make_pair(context->id, context));

		return context;
	}

	static WaterzContext* get(int id) {

		if (!_contexts.count(id))
			return NULL;

		return _contexts.at(id);
	}

	static void free(int id) {

		WaterzContext* context = get(id);

		if (context) {

			_contexts.erase(id);
			delete context;
		}
	}

	int id;

	std::shared_ptr<RegionGraphType> regionGraph;
	std::shared_ptr<RegionMergingType> regionMerging;
	std::shared_ptr<ScoringFunctionType> scoringFunction;
	std::shared_ptr<StatisticsProviderType> statisticsProvider;
	volume_ref_ptr<SegID> segmentation;
	volume_const_ref_ptr<GtID> groundtruth;
	UnmergeGroupListTupleList unmergeList;

private:

	WaterzContext() {}

	~WaterzContext() {}

	static std::map<int, WaterzContext*> _contexts;
	static int _nextId;
};

class UnmergeTracker {

public:

	using GroupId = SegID;
	using GroupIdList = std::vector<GroupId>;
	using AntiGroupListTuple = std::vector<GroupIdList>;
	// using UnmergeGroupListTupleList = std::vector<AntiGroupListTuple>;

	UnmergeTracker(const UnmergeGroupListTupleList& input_list) {

		/* The input_list is a list of tuples.
			Each tuple containing a variable number of coherent groupid's
				such that each segid in a coherent group cannot be merged
				with any other segid in another coherent group.
		*/

		if (input_list.size() == 0) {
			_is_empty = true;
			return;
		}

		for (const auto& anti_group_list_tuple : input_list) {
			// each anti_group_list_tuple is independent of each other
			GroupIdList mutex_group_id_list;
			for (const auto& segid_list : anti_group_list_tuple) {
				// segid_list contains a list of segid coherent with each other
				// we simply select the first element to represent as the groupid
				GroupId groupid = segid_list[0];
				// std::cout << "new groupid: " << groupid << std::endl << ": ";

				// add it to the mutually exclusive list
				mutex_group_id_list.push_back(groupid);
				// add this group id to each segid in the list
				for (auto segid : segid_list) {
					// std::cout << segid << " ";
					_segid_to_groupid_list[segid].push_back(groupid);
				}
				// std::cout << std::endl;
			}

			// update _groupid_to_anti_list with this tuple
			for (auto groupid : mutex_group_id_list) {
				GroupIdList others;
				for (auto other : mutex_group_id_list) {
					if (other != groupid)
						others.push_back(other);
				}

				// update
				auto& l = _groupid_to_anti_list[groupid];
				l.insert(std::end(l), std::begin(others), std::end(others));
			}
		}

	}

	bool isValidMerge(SegID a, SegID b) {
		if (_is_empty)
			return true;

		// std::cout << "isValidMerge " << a << " with " << b << std::endl;

		GroupIdList groups_a;
		GroupIdList groups_b;
		GroupIdList anti_groups_a;
		getGroupIDs(a, &groups_a);
		getGroupIDs(b, &groups_b);
		// std::cout << "groups_a: ";
		for (const auto& groupid : groups_a) {
			// std::cout << groupid;
		}
		// std::cout << std::endl;
		// std::cout << "groups_b: ";
		for (const auto& groupid : groups_b) {
			// std::cout << groupid;
		}
		// std::cout << std::endl;

		for (const auto& group_a : groups_a) {
			getAntiGroupIDs(group_a, &anti_groups_a);
			// std::cout << "anti_groups_a for " << group_a << ": ";
			for (const auto& groupid : anti_groups_a) {
				// std::cout << groupid;
			}
			for (const auto& group_b : groups_b) {
				for (const auto& anti_group_a : anti_groups_a) {
					if (anti_group_a == group_b) {
						// std::cout << "Cannot merge " << a << " with " << b << std::endl;
						return false;
					}
				}
			}
		}
		// checked and all good
		return true;
	}

	void onMerge(SegID a, SegID b, SegID c) {
		if (_is_empty)
			return;

		assert(c == a || c == b);

		GroupIdList groups_a;
		GroupIdList groups_b;
		getGroupIDs(a, &groups_a);
		getGroupIDs(b, &groups_b);

		groups_a.insert(std::end(groups_a),
			std::begin(groups_b), std::end(groups_b));

		if (groups_a.size())
			_segid_to_groupid_list[c] = groups_a;

		return;
	}

private:

	std::unordered_map<SegID, GroupIdList> _segid_to_groupid_list;
	std::unordered_map<GroupId, GroupIdList> _groupid_to_anti_list;
	bool _is_empty = false;

	void getGroupIDs(SegID segid, GroupIdList* ret) {
		if (_segid_to_groupid_list.count(segid)) {
			*ret = _segid_to_groupid_list[segid];
		}
	}

	void getAntiGroupIDs(SegID segid, GroupIdList* ret) {
		if (_groupid_to_anti_list.count(segid)) {
			*ret = _groupid_to_anti_list[segid];
		}
	}
};


class RegionMergingVisitor {

public:

	void onPop(RegionGraphType::EdgeIdType e, ScoreValue score) {}

	void onDeletedEdgeFound(RegionGraphType::EdgeIdType e) {}

	void onStaleEdgeFound(RegionGraphType::EdgeIdType e, ScoreValue oldScore, ScoreValue newScore) {}

	void onMerge(SegID a, SegID b, SegID c, ScoreValue score) {
		if (_unmergeTracker != NULL)
			_unmergeTracker->onMerge(a, b, c);
	}

	bool isValidMerge(SegID a, SegID b) {
		if (_unmergeTracker != NULL)
			return _unmergeTracker->isValidMerge(a, b);
		return true;
	}

	void setUnmergeTracker(UnmergeTracker* tracker) {
		_unmergeTracker = tracker;
	}

private:

	UnmergeTracker* _unmergeTracker = NULL;
};

class MergeHistoryVisitor : public RegionMergingVisitor {

public:

	MergeHistoryVisitor(std::vector<Merge>& history) : _history(history) {}

	void onMerge(SegID a, SegID b, SegID c, ScoreValue score) {

		_history.push_back({a, b, c, score});
		RegionMergingVisitor::onMerge(a, b, c, score);
	}

private:

	std::vector<Merge>& _history;
};

WaterzState initialize(
		size_t          width,
		size_t          height,
		size_t          depth,
		const AffValue* affinity_data,
		SegID*          segmentation_data,
		const GtID*     groundtruth_data = NULL,
		AffValue        affThresholdLow  = 0.0001,
		AffValue        affThresholdHigh = 0.9999,
		bool            findFragments = true,
		std::vector<std::vector<std::vector<uint64_t>>>* unmergeList = NULL);

std::vector<Merge> mergeUntil(
		WaterzState& state,
		float        threshold);

std::vector<ScoredEdge> getRegionGraph(WaterzState& state);

void free(WaterzState& state);

#endif
