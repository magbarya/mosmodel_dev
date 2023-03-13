#! /usr/bin/env python3

import pandas as pd
import numpy as np

class PerformanceStatistics:
    def __init__(self, csv_file, index_col=None):
        self._df = pd.read_csv(csv_file, index_col=index_col)
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

    def getIndexColumn(self):
        return np.array(self._df.index)

    def getWalkDuration(self, index=None):
        walk_duration = self.__getWalkCounter(index, 'active')
        if walk_duration is not None:
            return walk_duration
        walk_duration = self.__getWalkCounter(index, 'duration')
        if walk_duration is not None:
            return walk_duration
        raise Exception('the data-set has no performance counters for page walks!')

    def getStlbHits(self, index=None):
        data_set = self.__getDataSet(index)
        if 'dtlb_load_misses.stlb_hit' in self._df.columns \
                and 'dtlb_store_misses.stlb_hit' in self._df.columns:
                    return data_set['dtlb_load_misses.stlb_hit'] \
                            + data_set['dtlb_store_misses.stlb_hit']
        else:
            raise Exception('the data-set has no performance counters for STLB hits!')

    def getStlbMisses(self, index=None):
        try:
            return self.getStlbMissesRetired(index)
        except:
            pass
        try:
            return self.getStlbMissesCompleted(index)
        except:
            pass
        try:
            return self.getStlbMissesStarted(index)
        except:
            pass

        raise Exception('the data-set has no performance counters for TLB misses!')

    def getStlbMissesStarted(self, index=None):
        data_set = self.__getDataSet(index)
        if 'dtlb_load_misses.miss_causes_a_walk' in self._df.columns \
                and 'dtlb_store_misses.miss_causes_a_walk' in self._df.columns:
                    return data_set['dtlb_load_misses.miss_causes_a_walk'] \
                            + data_set['dtlb_store_misses.miss_causes_a_walk']
        else:
            raise Exception('the data-set has no performance counters for "started" TLB misses!')

    def getStlbMissesCompleted(self, index=None):
        data_set = self.__getDataSet(index)
        if 'dtlb_load_misses.walk_completed' in self._df.columns \
                and 'dtlb_store_misses.walk_completed' in self._df.columns:
                    return data_set['dtlb_load_misses.walk_completed'] \
                            + data_set['dtlb_store_misses.walk_completed']
        else:
            raise Exception('the data-set has no performance counters for "completed" TLB misses!')

    def getStlbMissesRetired(self, index=None):
        data_set = self.__getDataSet(index)
        if 'mem_inst_retired.stlb_miss_loads' in self._df.columns \
                and 'mem_inst_retired.stlb_miss_stores' in self._df.columns:
                    return data_set['mem_inst_retired.stlb_miss_loads'] \
                            + data_set['mem_inst_retired.stlb_miss_stores']
        else:
            raise Exception('the data-set has no performance counters for "retired" TLB misses!')

    def getStlbMisses2m(self, index=None):
        data_set = self.__getDataSet(index)
        if 'dtlb_load_misses.walk_completed_2m_4m' in self._df.columns \
        and 'dtlb_store_misses.walk_completed_2m_4m' in self._df.columns:
            return data_set['dtlb_load_misses.walk_completed_2m_4m'] \
                    + data_set['dtlb_store_misses.walk_completed_2m_4m']
        else:
            return self.getStlbMisses(index)
            #raise Exception('the data-set has no performance counters for STLB misses (for 2MB pages)!')

    def getStlbMisses4k(self, index=None):
        data_set = self.__getDataSet(index)
        if 'dtlb_load_misses.walk_completed_4k' in self._df.columns \
        and 'dtlb_store_misses.walk_completed_4k' in self._df.columns:
            return data_set['dtlb_load_misses.walk_completed_4k'] \
                    + data_set['dtlb_store_misses.walk_completed_4k']
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
        data_set = self.__getDataSet(index)
        if 'L1-dcache-loads' in self._df.columns \
                and 'L1-dcache-stores' in self._df.columns:
                    return data_set['L1-dcache-loads'] \
                            + data_set['L1-dcache-stores']
        else:
            raise Exception('the data-set has no performance counters for L1 data cache accesses!')

    def getL1Misses(self, index=None):
        data_set = self.__getDataSet(index)
        if 'L1-dcache-load-misses' in self._df.columns \
                and 'L1-dcache-store-misses' in self._df.columns:
                    return data_set['L1-dcache-load-misses'] \
                            + data_set['L1-dcache-store-misses']
        else:
            raise Exception('the data-set has no performance counters for L1 data cache misses!')

    def getL1Hits(self, index=None):
        misses = self.getL1Misses(index)
        accesses = self.getL1Accesses(index)
        hits = accesses - misses
        return hits

    def getLlcAccesses(self, index=None):
        data_set = self.__getDataSet(index)
        if 'LLC-loads' in self._df.columns \
                and 'LLC-stores' in self._df.columns:
                    return data_set['LLC-loads'] \
                            + data_set['LLC-stores']
        else:
            raise Exception('the data-set has no performance counters for LLC accesses!')

    def getLlcMisses(self, index=None):
        data_set = self.__getDataSet(index)
        if 'LLC-load-misses' in self._df.columns\
                and 'LLC-store-misses' in self._df.columns:
                    return data_set['LLC-load-misses'] \
                            + data_set['LLC-store-misses']
        else:
            raise Exception('the data-set has no performance counters for LLC misses!')

    def getLlcHits(self, index=None):
        return self.getLlcAccesses(index) - self.getLlcMisses(index)

    def getL2Accesses(self, index=None):
        return self.getL1Misses(index)

    def getL2Misses(self, index=None):
        return self.getLlcAccesses(index)

    def getL2Hits(self, index=None):
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
