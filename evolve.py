from utils import StatusUpdateTool, Utils, Log
from genetic.population import Population
from genetic.evaluate import FitnessEvaluate
from genetic.crossover_and_mutation import CrossoverAndMutation
from genetic.selection_operator import Selection
import numpy as np
import copy

class EvolveCNN(object):
    def __init__(self, params):
        self.params = params
        self.pops = None

    def initialize_population(self):
        StatusUpdateTool.begin_evolution()
        pops = Population(params, 0)
        pops.initialize()
        self.pops = pops
        Utils.save_population_at_begin(str(pops), 0)

    def fitness_evaluate(self):
        fitness = FitnessEvaluate(self.pops.individuals, Log)
        fitness.generate_to_python_file()
        fitness.evaluate()


    def crossover_and_mutation(self):
        cm = CrossoverAndMutation(self.params['genetic_prob'][0], self.params['genetic_prob'][1], Log, self.pops.individuals, _params={'gen_no': self.pops.gen_no})
        offspring = cm.process()
        self.parent_pops = copy.deepcopy(self.pops)
        self.pops.individuals = copy.deepcopy(offspring)

    def environment_selection(self):
        v_list = []
        indi_list = []
        for indi in self.pops.individuals:
            indi_list.append(indi)
            v_list.append(indi.acc)
        for indi in self.parent_pops.individuals:
            indi_list.append(indi)
            v_list.append(indi.acc)

        _str = []
        for _, indi in enumerate(self.pops.individuals):
            _t_str = 'Indi-%s-%.5f-%s'%(indi.id, indi.acc, indi.uuid()[0])
            _str.append(_t_str)
        for _, indi in enumerate(self.parent_pops.individuals):
            _t_str = 'Pare-%s-%.5f-%s'%(indi.id, indi.acc, indi.uuid()[0])
            _str.append(_t_str)


        #add log
        # find the largest one's index
        max_index = np.argmax(v_list)
        selection = Selection()
        selected_index_list = selection.RouletteSelection(v_list, k=self.params['pop_size'])
        if max_index not in selected_index_list:
            first_selectd_v_list = [v_list[i] for i in selected_index_list]
            min_idx = np.argmin(first_selectd_v_list)
            selected_index_list[min_idx] = max_index

        next_individuals = [indi_list[i] for i in selected_index_list]

        """Here, the population information should be updated, such as the gene no and then to the individual id"""
        next_gen_pops = Population(self.pops.params, self.pops.gen_no+1)
        next_gen_pops.create_from_offspring(next_individuals)
        self.pops = next_gen_pops
        for _, indi in enumerate(self.pops.individuals):
            _t_str = 'new -%s-%.5f-%s'%(indi.id, indi.acc, indi.uuid()[0])
            _str.append(_t_str)
        _file = './populations/ENVI_%2d.txt'%(self.pops.gen_no)
        Utils.write_to_file('\n'.join(_str), _file)

        Utils.save_population_at_begin(str(self.pops), self.pops.gen_no)

    def do_work(self, max_gen):
        Log.info('*'*25)
        # the step 1
        if StatusUpdateTool.is_evolution_running():
            Log.info('Initialize from existing population data')
            gen_no = Utils.get_newest_file_based_on_prefix('begin')
            if gen_no is not None:
                Log.info('Initialize from %d-th generation'%(gen_no))
                pops = Utils.load_population('begin', gen_no)
                self.pops = pops
            else:
                raise ValueError('The running flag is set to be running, but there is no generated population stored')
        else:
            gen_no = 0
            Log.info('Initialize...')
            self.initialize_population()
        Log.info('EVOLVE[%d-gen]-Begin to evaluate the fitness'%(gen_no))
        self.fitness_evaluate()
        Log.info('EVOLVE[%d-gen]-Finish the evaluation'%(gen_no))
        gen_no += 1
        for curr_gen in range(gen_no, max_gen):
            self.params['gen_no'] = curr_gen
            #step 3
            Log.info('EVOLVE[%d-gen]-Begin to crossover and mutation'%(curr_gen))
            self.crossover_and_mutation()
            Log.info('EVOLVE[%d-gen]-Finish crossover and mutation'%(curr_gen))

            Log.info('EVOLVE[%d-gen]-Begin to evaluate the fitness'%(curr_gen))
            self.fitness_evaluate()
            Log.info('EVOLVE[%d-gen]-Finish the evaluation'%(curr_gen))

            self.environment_selection()
            Log.info('EVOLVE[%d-gen]-Finish the environment selection'%(curr_gen))

        StatusUpdateTool.end_evolution()
if __name__ == '__main__':
    params = StatusUpdateTool.get_init_params()
    evoCNN = EvolveCNN(params)
    evoCNN.do_work(max_gen=20)



