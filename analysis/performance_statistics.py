#! /usr/bin/env python3

import pandas as pd
import numpy as np

class PerformanceStatistics:
    def __init__(self, perf_file, index_col=None):
        if type(perf_file) is str:
            self._df = pd.read_csv(perf_file, index_col=index_col)
        elif type(perf_file) is pd.DataFrame:
            self._df = perf_file.copy()
        else:
            assert False
        self._df.fillna(0, inplace=True)

    def __getDataSet(self, index=None):
        if index==None:
            return self._df
        else:
            return self._df.loc[index]

    def __getWalkCounter(self, index, counter_suffix):
        data_set = self.__getDataSet(index)
        load_counter = 'dtlb_load_misses.walk_' + counter_suffix
        store_counter = 'dtlb_store_misses.walk_' + counter_suffix
        if load_counter in self._df.columns \
                and store_counter in self._df.columns:
                    return data_set[load_counter] + data_set[store_counter]
        else:
            return None

    def __getLoadStoreCountersSum(self, index, load_counter, store_counter):
        data_set = self.__getDataSet(index)
        if load_counter in self._df.columns \
                and store_counter in self._df.columns:
                    return data_set[load_counter] + data_set[store_counter]
        else:
            return None

    def __getCounter(self, index, counter):
        data_set = self.__getDataSet(index)
        if counter in self._df.columns:
            return data_set[counter]
        else:
            return None

    def getIndexColumn(self):
        return np.array(self._df.index)

    def getWalkPendingCycles(self, index=None):
        walk_cycles = self.__getWalkCounter(index, 'pending')
        if walk_cycles is not None:
            return walk_cycles
        raise Exception('the data-set has no performance counters for walk pending!')

    def getWalkActiveCycles(self, index=None):
        walk_cycles = self.__getWalkCounter(index, 'active')
        if walk_cycles is not None:
            return walk_cycles
        raise Exception('the data-set has no performance counters for walk active!')

    def getWalkDuration(self, index=None):
        walk_duration = self.__getWalkCounter(index, 'pending')
        if walk_duration is not None:
            return walk_duration
        walk_duration = self.__getWalkCounter(index, 'active')
        if walk_duration is not None:
            return walk_duration
        walk_duration = self.__getWalkCounter(index, 'duration')
        if walk_duration is not None:
            return walk_duration
        raise Exception('the data-set has no performance counters for page walks!')

    def getStlbHits(self, index=None):
        load_counter = 'dtlb_load_misses.stlb_hit'
        store_counter = 'dtlb_store_misses.stlb_hit'
        val = self.__getLoadStoreCountersSum(index, load_counter, store_counter)
        if val is not None:
            return val
        else:
            raise Exception('the data-set has no performance counters for STLB hits!')

    def getStlbMisses(self, index=None):
        return self.getStlbMisses_completed(index)

    def getStlbMisses_started(self, index=None):
        load_counter = 'dtlb_load_misses.miss_causes_a_walk'
        store_counter = 'dtlb_store_misses.miss_causes_a_walk'
        val = self.__getLoadStoreCountersSum(index, load_counter, store_counter)
        if val is not None:
            return val
        else:
            raise Exception('the data-set has no performance counters for TLB misses!')

    def getStlbMisses_completed(self, index=None):
        load_counter = 'dtlb_load_misses.walk_completed'
        store_counter = 'dtlb_store_misses.walk_completed'
        val = self.__getLoadStoreCountersSum(index, load_counter, store_counter)
        if val is not None:
            return val
        else:
            raise Exception('the data-set has no performance counters for STLB misses (dtlb-misses-walk-completed)!')

    def getStlbMisses2m_completed(self, index=None):
        load_counter = 'dtlb_load_misses.walk_completed_2m_4m'
        store_counter = 'dtlb_store_misses.walk_completed_2m_4m'
        val = self.__getLoadStoreCountersSum(index, load_counter, store_counter)
        if val is not None:
            return val
        else:
            return self.getStlbMisses(index)
            #raise Exception('the data-set has no performance counters for STLB misses (for 2MB pages)!')

    def getStlbMisses4k_completed(self, index=None):
        load_counter = 'dtlb_load_misses.walk_completed_4k'
        store_counter = 'dtlb_store_misses.walk_completed_4k'
        val = self.__getLoadStoreCountersSum(index, load_counter, store_counter)
        if val is not None:
            return val
        else:
            return self.getStlbMisses(index)
            #raise Exception('the data-set has no performance counters for STLB misses (for 4KB pages)!')

    def getStlbAccesses(self, index=None):
        return self.getStlbHits(index) + self.getStlbMisses(index)

    def getTlbAccesses(self, index=None):
        return self.getL1Accesses(index)

    def getTlbMisses(self, index=None):
        return self.getStlbAccesses(index)

    def getTlbHits(self, index=None):
        return self.getTlbAccesses(index) - self.getTlbMisses(index)

    def getL1Accesses(self, index=None):
        load_counter = 'L1-dcache-loads'
        store_counter = 'L1-dcache-stores'
        val = self.__getLoadStoreCountersSum(index, load_counter, store_counter)
        if val is not None:
            return val
        else:
            raise Exception('the data-set has no performance counters for L1 data cache accesses!')

    def getL1Misses(self, index=None):
        val = self.__getLoadStoreCountersSum(index, 'L1-dcache-load-misses', 'L1-dcache-store-misses')
        if val is not None:
            return val
        val = self.__getCounter(index, 'mem_load_retired.l1_miss')
        if val is not None:
            return val
        else:
            raise Exception('the data-set has no performance counters for L1 data cache misses!')

    def getL1Hits(self, index=None):
        val = self.__getLoadStoreCountersSum(index, 'L1-dcache-load-hits', 'L1-dcache-store-hits')
        if val is not None:
            return val
        val = self.__getCounter(index, 'mem_load_retired.l1_hit')
        if val is not None:
            return val
        misses = self.getL1Misses(index)
        accesses = self.getL1Accesses(index)
        hits = accesses - misses
        return hits

    def getLlcAccesses(self, index=None):
        load_counter = 'LLC-loads'
        store_counter = 'LLC-stores'
        val = self.__getLoadStoreCountersSum(index, load_counter, store_counter)
        if val is not None:
            return val
        else:
            raise Exception('the data-set has no performance counters for LLC accesses!')

    def getLlcMisses(self, index=None):
        load_counter = 'LLC-load-misses'
        store_counter = 'LLC-store-misses'
        val = self.__getLoadStoreCountersSum(index, load_counter, store_counter)
        if val is not None:
            return val
        val = self.__getCounter(index, 'mem_load_retired.l3_miss')
        if val is not None:
            return val
        else:
            raise Exception('the data-set has no performance counters for LLC misses!')

    def getLlcHits(self, index=None):
        val = self.__getCounter(index, 'mem_load_retired.l3_hit')
        if val is not None:
            return val
        return self.getLlcAccesses(index) - self.getLlcMisses(index)

    def getL2Accesses(self, index=None):
        return self.getL1Misses(index)

    def getL2Misses(self, index=None):
        val = self.__getCounter(index, 'mem_load_retired.l2_miss')
        if val is not None:
            return val
        return self.getLlcAccesses(index)

    def getL2Hits(self, index=None):
        val = self.__getCounter(index, 'mem_load_retired.l2_hit')
        if val is not None:
            return val
        return self.getL2Accesses(index) - self.getL2Misses(index)

    def getPageWalkerL1Hits(self, index=None):
        data_set = self.__getDataSet(index)
        if 'page_walker_loads.dtlb_l1' in self._df.columns:
            return data_set['page_walker_loads.dtlb_l1']
        else:
            raise Exception('the data-set has no performance counters for page_walker_loads.dtlb_l1!')

    def getPageWalkerL2Hits(self, index=None):
        data_set = self.__getDataSet(index)
        if 'page_walker_loads.dtlb_l2' in self._df.columns:
            return data_set['page_walker_loads.dtlb_l2']
        else:
            raise Exception('the data-set has no performance counters for page_walker_loads.dtlb_l2!')

    def getPageWalkerL3Hits(self, index=None):
        data_set = self.__getDataSet(index)
        if 'page_walker_loads.dtlb_l3' in self._df.columns:
            return data_set['page_walker_loads.dtlb_l3']
        else:
            raise Exception('the data-set has no performance counters for page_walker_loads.dtlb_l3!')

    def getPageWalkerMemoryAccesses(self, index=None):
        data_set = self.__getDataSet(index)
        if 'page_walker_loads.dtlb_memory' in self._df.columns:
            return data_set['page_walker_loads.dtlb_memory']
        else:
            raise Exception('the data-set has no performance counters for page_walker_loads.dtlb_memory!')


    def getRuntime(self, index=None):
        data_set = self.__getDataSet(index)
        if 'cpu-cycles' in self._df.columns:
            return data_set['cpu-cycles']
        else:
            raise Exception('the data-set has no performance counters for CPU cycles!')
        return 0

    def getRefCycles(self, index=None):
        data_set = self.__getDataSet(index)
        if self._df.columns.contains('ref-cycles'):
            return data_set['ref-cycles']
        else:
            raise Exception('the data-set has no performance counters for ref cycles!')
        return 0

    def getDataFrame(self):
        return self._df.copy()
