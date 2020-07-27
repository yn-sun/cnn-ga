
"""
number of conv/pool
                    ----add conv/pool
                    ----remove conv/pool
properties of conv
                    ----kerner size of conv
                    ----pooling type
                    ----connections

for each individual, use 5 bits to denotes such five operations then do bit-wise flip, if flip, then the operation is done
for each operation
                ---add conv/pool    | use bits to denote the units, then do bit-wise flip to determine where to add
                ---remove conv/pool | the same to the above
"""
import random
import numpy as np
import copy
from utils import StatusUpdateTool, Utils

class CrossoverAndMutation(object):
    def __init__(self, prob_crossover, prob_mutation, _log, individuals, _params=None):
        self.prob_crossover = prob_crossover
        self.prob_mutation = prob_mutation
        self.individuals = individuals
        self.params = _params # storing other parameters if needed, such as the index for SXB and polynomial mutation
        self.log = _log
        self.offspring = []

    def process(self):
        crossover = Crossover(self.individuals, self.prob_crossover, self.log)
        offspring = crossover.do_crossover()
        self.offspring = offspring
        Utils.save_population_after_crossover(self.individuals_to_string(), self.params['gen_no'])

        mutation = Mutation(self.offspring, self.prob_mutation, self.log)
        mutation.do_mutation()

        for i, indi in enumerate(self.offspring):
            indi_no = 'indi%02d%02d'%(self.params['gen_no'], i)
            indi.id = indi_no

        Utils.save_population_after_mutation(self.individuals_to_string(), self.params['gen_no'])
        return offspring

    def individuals_to_string(self):
        _str = []
        for ind in self.offspring:
            _str.append(str(ind))
            _str.append('-'*100)
        return '\n'.join(_str)



class Crossover(object):
    def __init__(self, individuals, prob_, _log):
        self.individuals = individuals
        self.prob = prob_
        self.log = _log
        self.pool_limit = StatusUpdateTool.get_pool_limit()[1]

    def _choose_one_parent(self):
        count_ = len(self.individuals)
        idx1 = int(np.floor(np.random.random()*count_))
        idx2 = int(np.floor(np.random.random()*count_))
        while idx2 == idx1:
            idx2 = int(np.floor(np.random.random()*count_))

        if self.individuals[idx1].acc > self.individuals[idx2].acc:
            return idx1
        else:
            return idx2
    """
    binary tournament selection
    """
    def _choose_two_diff_parents(self):
        idx1 = self._choose_one_parent()
        idx2 = self._choose_one_parent()
        while idx2 == idx1:
            idx2 = self._choose_one_parent()

        assert idx1 < len(self.individuals)
        assert idx2 < len(self.individuals)
        return idx1, idx2

    """
    calculate the number of pooling units after the crossover is done
    """
    def _calculate_pool_numbers(self, parent1, parent2):
        t1, t2 = 0, 0
        for unit in parent1.units:
            if unit.type == 2:
                t1 += 1
        for unit in parent2.units:
            if unit.type == 2:
                t2 += 1

        len1, len2 = len(parent1.units), len(parent2.units)
        pos1, pos2 = int(np.floor(np.random.random()*len1)), int(np.floor(np.random.random()*len2))
        assert pos1 < len1
        assert pos2 < len2
        p1_left, p1_right, p2_left, p2_right = 0, 0, 0, 0
        for i in range(0, pos1):
            if parent1.units[i].type == 2:
                p1_left += 1
        for i in range(pos1, len1):
            if parent1.units[i].type == 2:
                p1_right += 1

        for i in range(0,pos2):
            if parent2.units[i].type == 2:
                p2_left += 1
        for i in range(pos2, len2):
            if parent2.units[i].type == 2:
                p2_right += 1

        new_pool_number1 = p1_left + p2_right
        new_pool_number2 = p2_left + p1_right
        return pos1, pos2, new_pool_number1,new_pool_number2

    def do_crossover(self):
        _stat_param = {'offspring_new':0, 'offspring_from_parent':0}
        new_offspring_list = []
        for _ in range(len(self.individuals)//2):
            ind1, ind2 = self._choose_two_diff_parents()
            parent1, parent2 = copy.deepcopy(self.individuals[ind1]), copy.deepcopy(self.individuals[ind2])
            p_ = random.random()
            if p_ < self.prob:
                _stat_param['offspring_new'] += 2
                """
                exchange their units from these parent individuals, the exchanged units must satisfy
                --- the number of pooling layer should not be more than the predefined setting
                --- if their is no changing after this crossover, keep the original acc -- a mutation should be given [to do---]
                """
                first_begin_is_pool, second_begin_is_pool = True, True
                while first_begin_is_pool is True or second_begin_is_pool is True:
                    pos1, pos2, pool_len1, pool_len2 = self._calculate_pool_numbers(parent1, parent2)
                    try_count = 1
                    while pool_len1 > self.pool_limit or pool_len2 > self.pool_limit:
                        pos1, pos2, pool_len1, pool_len2 = self._calculate_pool_numbers(parent1, parent2)
                        try_count += 1
                        self.log.warn('The %d-th try to find the position for do crossover'%(try_count))
                    self.log.info('Position %d for %s, positions %d for %s'%(pos1, parent1.id, pos2, parent2.id))
                    unit_list1, unit_list2 = [], []
                    for i in range(0, pos1):
                        unit_list1.append(parent1.units[i])
                    for i in range(pos2, len(parent2.units)):
                        unit_list1.append(parent2.units[i])

                    for i in range(0, pos2):
                        unit_list2.append(parent2.units[i])
                    for i in range(pos1, len(parent1.units)):
                        unit_list2.append(parent1.units[i])
                    first_begin_is_pool = True if unit_list1[0].type == 2 else False
                    second_begin_is_pool = True if unit_list2[0].type == 2 else False

                    if first_begin_is_pool is True:
                        self.log.warn('Crossovered individual#1 starts with a pooling layer, redo...')
                    if second_begin_is_pool is True:
                        self.log.warn('Crossovered individual#2 starts with a pooling layer, redo...')



                # reorder the number of each unit based on its order in the list
                for i, unit in enumerate(unit_list1):
                    unit.number = i
                for i, unit in enumerate(unit_list2):
                    unit.number = i

                # re-adjust the in_channel of the next conv layer

                last_output_from_list1 = 0
                if pos1 == 0:
                    last_output_from_list1 = StatusUpdateTool.get_input_channel()
                    j = 0
                    i = -1
                else:
                    for i in range(pos1-1, -1, -1):
                        if unit_list1[i].type == 1:
                            last_output_from_list1 = unit_list1[i].out_channel
                            break
                for j in range(pos1, len(unit_list1)):
                    if unit_list1[j].type == 1:
                        unit_list1[j].in_channel = last_output_from_list1
                        break
                self.log.info('Change the input channel of unit at %d to %d that is the output channel of unit at %d in %s'%(j, last_output_from_list1, i, parent1.id))

                last_output_from_list2 = 0

                if pos2 == 0:
                    last_output_from_list2 = StatusUpdateTool.get_input_channel()
                    j = 0
                    i = -1
                else:
                    for i in range(pos2-1, -1, -1):
                        if unit_list2[i].type == 1:
                            last_output_from_list2 = unit_list2[i].out_channel
                            break
                for j in range(pos2, len(unit_list2)):
                    if unit_list2[j].type == 1:
                        unit_list2[j].in_channel = last_output_from_list2
                        break
                self.log.info('Change the input channel of unit at %d to %d that is the output channel of unit at %d in %s'%(j, last_output_from_list2, i, parent2.id))


                parent1.units = unit_list1
                parent2.units = unit_list2
                offspring1, offspring2 = parent1, parent2
                offspring1.reset_acc()
                offspring2.reset_acc()
                new_offspring_list.append(offspring1)
                new_offspring_list.append(offspring2)
            else:
                _stat_param['offspring_from_parent'] += 2
                new_offspring_list.append(parent1)
                new_offspring_list.append(parent2)

        self.log.info('CROSSOVER-%d offspring are generated, new:%d, others:%d'%(len(new_offspring_list), _stat_param['offspring_new'],_stat_param['offspring_from_parent']))
        return new_offspring_list



class Mutation(object):

    def __init__(self, individuals, prob_, _log):
        self.individuals = individuals
        self.prob = prob_
        self.log = _log

    def do_mutation(self):
        _stat_param = {'offspring_new':0, 'offspring_from_parent':0, 'ADD':0, 'REMOVE':0, 'CHANNEL':0, 'POOLING_TYPE':0}

        mutation_list = StatusUpdateTool.get_mutation_probs_for_each()
        for indi in self.individuals:
            p_ = random.random()
            if p_ < self.prob:
                _stat_param['offspring_new'] += 1
                mutation_type = self.select_mutation_type(mutation_list)
                if mutation_type == 0:
                    _stat_param['ADD'] += 1
                    self.do_add_unit_mutation(indi)
                elif mutation_type == 1:
                    _stat_param['REMOVE'] += 1
                    self.do_remove_unit_mutation(indi)
                elif mutation_type == 2:
                    _stat_param['CHANNEL'] += 1
                    self.do_modify_conv_mutation(indi)
                elif mutation_type == 3:
                    _stat_param['POOLING_TYPE'] += 1
                    self.do_modify_pooling_type_mutation(indi)
                else:
                    raise TypeError('Error mutation type :%d, validate range:0-4'%(mutation_type))
            else:
                _stat_param['offspring_from_parent'] += 1
        self.log.info('MUTATION-mutated individuals:%d[ADD:%2d,REMOVE:%2d,CHANNEL:%2d,POOL:%2d, no_changes:%d'%(_stat_param['offspring_new'], \
                      _stat_param['ADD'], _stat_param['REMOVE'], _stat_param['CHANNEL'], _stat_param['POOLING_TYPE'],  _stat_param['offspring_from_parent']))



    def do_add_unit_mutation(self, indi):
        self.log.info('Do the ADD mutation for indi:%s'%(indi.id))
        """
        choose one position to add one unit, adding one conv or pooling unit is determined by a probability of 0.5.
        However, if the maximal number of pooling units have been added into the current individual, only
        conv unit will be add here
        """
        # determine the position where a unit would be added
        mutation_position = int(np.floor(np.random.random()*len(indi.units)))
        self.log.info('Mutation position occurs at %d'%(mutation_position))
        # determine the unit type for adding
        u_ = random.random()
        type_ = 1 if u_ < 0.5 else 2
        self.log.info('A %s unit would be added due to the probability of %.2f'%('CONV' if type_ ==1 else 'POOLING', u_))
        if type_ == 2:
            num_exist_pool_units = 0
            for unit in indi.units:
                if unit.type == 2:
                    num_exist_pool_units +=1
            if num_exist_pool_units > StatusUpdateTool.get_pool_limit()[1]-1:
                type_ = 1
                self.log.info('The added unit is changed to CONV because the existing number of POOLING exceeds %d, limit size:%d'%(num_exist_pool_units, StatusUpdateTool.get_pool_limit()[1]))

        #do the details
        if type_ == 2:
            add_unit = indi.init_a_pool(mutation_position+1, _max_or_avg=None)
        else:
            for i in range(mutation_position, -1, -1):
                if indi.units[i].type == 1:
                    _in_channel = indi.units[i].out_channel
                    break
            add_unit = indi.init_a_conv(mutation_position+1, _in_channel=_in_channel, _out_channel=None)
            for i in range(mutation_position+1, len(indi.units)):
                if indi.units[i].type == 1:
                    indi.units[i].in_channel = add_unit.out_channel
                    break

        new_unit_list = []
        # add to the new list and update the number
        for i in range(mutation_position+1):
            new_unit_list.append(indi.units[i])
        new_unit_list.append(add_unit)
        for i in range(mutation_position+1, len(indi.units)):
            unit = indi.units[i]
            unit.number += 1
            new_unit_list.append(unit)
        indi.number_id += 1
        indi.units = new_unit_list
        indi.reset_acc()

    def do_remove_unit_mutation(self, indi):
        self.log.info('Do the REMOVE mutation for indi:%s'%(indi.id))
        if len(indi.units) > 1:
            mutation_position = int(np.floor(np.random.random()*(len(indi.units)-1))) + 1 # the first unit would not be removed
            self.log.info('Mutation position occurs at %d'%(mutation_position))
            if indi.units[mutation_position].type == 1:
                for i in range(mutation_position, -1, -1):
                    if indi.units[i].type == 1:
                        indi.units[i].out_channel = indi.units[mutation_position].out_channel
            new_unit_list = []
            for i in range(mutation_position):
                new_unit_list.append(indi.units[i])
            for i in range(mutation_position+1, len(indi.units)):
                unit = indi.units[i]
                unit.number -= 1
                new_unit_list.append(unit)
            indi.number_id -= 1
            indi.units = new_unit_list
            indi.reset_acc()
        else:
            self.log.warn('REMOVE mutation can not be performed due to it has only one unit')


    def do_modify_conv_mutation(self, indi):
        self.log.info('Do the CHANNEL mutation for indi:%s'%(indi.id))
        conv_index_list = []
        for i, unit in enumerate(indi.units):
            if unit.type == 1:
                conv_index_list.append(i)
        if len(conv_index_list) == 0:
            self.log.warn('No CONV unit exist in current individual, no mutation occurs')
        else:
            selected_index = int(np.floor(np.random.rand()*len(conv_index_list)))
            self.log.info('Mutation position %d'%(conv_index_list[selected_index]))

            channel_list = StatusUpdateTool().get_output_channel()
            index_ = int(np.floor(np.random.random()*len(channel_list)))
            if indi.units[conv_index_list[selected_index]].in_channel != channel_list[index_]:
                indi.reset_acc()
                if selected_index > 0:
                    self.log.info('Unit at %d changes its input channel from %d to %d'%(conv_index_list[selected_index], indi.units[conv_index_list[selected_index]].in_channel, channel_list[index_]))
                    indi.units[conv_index_list[selected_index]].in_channel = channel_list[index_]
                    self.log.info('Due to above, the unit at %d should change its output channel from %d to %d'%(conv_index_list[selected_index-1], indi.units[conv_index_list[selected_index-1]].out_channel, channel_list[index_]))
                    indi.units[conv_index_list[selected_index-1]].out_channel = channel_list[index_]
                else:
                    self.log.warn('Mutation position is 0, the input channel should not be changed')
            else:
                self.log.info('Unit at %d changes its input channel from %d to %d'%(conv_index_list[selected_index], indi.units[conv_index_list[selected_index]].in_channel, channel_list[index_]))

            index_ = int(np.floor(np.random.random()*len(channel_list)))
            if indi.units[conv_index_list[selected_index]].out_channel != channel_list[index_]:
                indi.reset_acc()
                self.log.info('Unit at %d changes its out channel from %d to %d'%(conv_index_list[selected_index], indi.units[conv_index_list[selected_index]].out_channel, channel_list[index_]))
                indi.units[conv_index_list[selected_index]].out_channel = channel_list[index_]
                if selected_index < len(conv_index_list)-1:
                    self.log.info('Due to above, the unit at %d should change its input channel from %d to %d'%(conv_index_list[selected_index+1], indi.units[conv_index_list[selected_index+1]].in_channel, channel_list[index_]))
                    indi.units[conv_index_list[selected_index+1]].in_channel = channel_list[index_]
                else:
                    self.log.info('Unit at %d is the last unit in the individual, therefore no need to change the input channel of the next unit')
            else:
                self.log.info('Unit at %d changes its out channel from %d to %d'%(conv_index_list[selected_index], indi.units[conv_index_list[selected_index]].out_channel, channel_list[index_]))

    def do_modify_pooling_type_mutation(self, indi):
        self.log.info('Do the POOLING TYPE mutation for indi:%s'%(indi.id))
        pool_list_index = []
        for i, unit in enumerate(indi.units):
            if unit.type == 2:
                pool_list_index.append(i)
        if len(pool_list_index) == 0:
            self.log.warn('No POOL unit exist, no mutation occurs')
        else:
            selected_index = int(np.floor(np.random.random()*len(pool_list_index)))
            self.log.info('Mutation position %d '%(pool_list_index[selected_index]))
            if indi.units[pool_list_index[selected_index]].max_or_avg > 0.5:
                indi.units[pool_list_index[selected_index]].max_or_avg = 0.2
                self.log.info('Pool type from avg_pool (>0.5) to max_pool (<0.5)')
            else:
                indi.units[pool_list_index[selected_index]].max_or_avg = 0.75
                self.log.info('Pool type from max_pool (<0.5) to avg_pool (>0.5)')
            indi.reset_acc()


    def select_mutation_type(self, _a):
        a = np.asarray(_a)
        k = 1
        idx = np.argsort(a)
        idx = idx[::-1]
        sort_a = a[idx]
        sum_a = np.sum(a).astype(np.float)
        selected_index = []
        for i in range(k):
            u = np.random.rand()*sum_a
            sum_ = 0
            for i in range(sort_a.shape[0]):
                sum_ +=sort_a[i]
                if sum_ > u:
                    selected_index.append(idx[i])
                    break
        return selected_index[0]


if __name__ == '__main__':
    m = Mutation(None, None, None)
    m.do_mutation()
