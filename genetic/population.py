import numpy as np
import hashlib
import copy

class Unit(object):
    def __init__(self, number):
        self.number = number


class ResUnit(Unit):
    def __init__(self, number, in_channel, out_channel): #prob < 0.5
        super().__init__(number)
        self.type = 1
        self.in_channel = in_channel
        self.out_channel = out_channel


class PoolUnit(Unit):
    def __init__(self, number, max_or_avg):
        super().__init__(number)
        self.type = 2
        self.max_or_avg = max_or_avg #max_pool for < 0.5 otherwise avg_pool


class Individual(object):
    def __init__(self, params, indi_no):
        self.acc = -1.0
        self.id = indi_no # for record the id of current individual
        self.number_id = 0 # for record the latest number of basic unit
        self.min_conv = params['min_conv']
        self.max_conv = params['max_conv']
        self.min_pool = params['min_pool']
        self.max_pool = params['max_pool']
        self.max_len = params['max_len']
        self.image_channel = params['image_channel']
        self.output_channles = params['output_channel']
        self.units = []

    def reset_acc(self):
        self.acc = -1.0

    def initialize(self):
        # initialize how many convolution and pooling layers will be used
        num_conv = np.random.randint(self.min_conv , self.max_conv+1)
        num_pool = np.random.randint(self.min_pool , self.max_pool+1)
        # find the position where the pooling layer can be connected
        availabel_positions = list(range(num_conv))
        np.random.shuffle(availabel_positions)
        select_positions = np.sort(availabel_positions[0:num_pool])
        all_positions = []
        for i in range(num_conv):
            all_positions.append(1) # 1 denotes the convolution layer, and 2 denotes the pooling layer
            for j in select_positions:
                if j == i:
                    all_positions.append(2)
                    break
        # initialize the layers based on their positions
        input_channel = self.image_channel
        for i in all_positions:
            if i == 1:
                conv = self.init_a_conv(_number=None,_in_channel=input_channel, _out_channel=None)
                input_channel = conv.out_channel
                self.units.append(conv)
            elif i == 2:
                pool = self.init_a_pool(_number=None, _max_or_avg=None)
                self.units.append(pool)

    """
    Initialize a convolutional layer
    """
    def init_a_conv(self, _number, _in_channel, _out_channel):
        if _number:
            number = _number
        else:
            number = self.number_id
            self.number_id += 1

        if _out_channel:
            out_channel = _out_channel
        else:
            out_channel = self.output_channles[np.random.randint(0, len(self.output_channles))]
        conv = ResUnit(number, _in_channel, out_channel)
        return conv

    def init_a_pool(self, _number, _max_or_avg):
        if _number:
            number = _number
        else:
            number = self.number_id
            self.number_id += 1

        if _max_or_avg:
            max_or_avg = _max_or_avg
        else:
            max_or_avg = np.random.rand()

        pool = PoolUnit(number, max_or_avg)
        return pool


    def uuid(self):
        _str = []
        for unit in self.units:
            _sub_str = []
            if unit.type == 1:
                _sub_str.append('conv')
                _sub_str.append('number:%d'%(unit.number))
                _sub_str.append('in:%d'%(unit.in_channel))
                _sub_str.append('out:%d'%(unit.out_channel))

            if unit.type == 2:
                _sub_str.append('pool')
                _sub_str.append('number:%d'%(unit.number))
                _pool_type = 0.25 if unit.max_or_avg < 0.5 else 0.75
                _sub_str.append('type:%.1f'%(_pool_type))
            _str.append('%s%s%s'%('[', ','.join(_sub_str), ']'))
        _final_str_ = '-'.join(_str)
        _final_utf8_str_= _final_str_.encode('utf-8')
        _hash_key = hashlib.sha224(_final_utf8_str_).hexdigest()
        return _hash_key, _final_str_

    def __str__(self):
        _str = []
        _str.append('indi:%s'%(self.id))
        _str.append('Acc:%.5f'%(self.acc))
        for unit in self.units:
            _sub_str = []
            if unit.type == 1:
                _sub_str.append('conv')
                _sub_str.append('number:%d'%(unit.number))
                _sub_str.append('in:%d'%(unit.in_channel))
                _sub_str.append('out:%d'%(unit.out_channel))

            if unit.type == 2:
                _sub_str.append('pool')
                _sub_str.append('number:%d'%(unit.number))
                _sub_str.append('type:%.1f'%(unit.max_or_avg))
            _str.append('%s%s%s'%('[', ','.join(_sub_str), ']'))
        return '\n'.join(_str)

class Population(object):
    def __init__(self, params, gen_no):
        self.gen_no = gen_no
        self.number_id = 0 # for record how many individuals have been generated
        self.pop_size = params['pop_size']
        self.params = params
        self.individuals = []

    def initialize(self):
        for _ in range(self.pop_size):
            indi_no = 'indi%02d%02d'%(self.gen_no, self.number_id)
            self.number_id += 1
            indi = Individual(self.params, indi_no)
            indi.initialize()
            self.individuals.append(indi)

    def create_from_offspring(self, offsprings):
        for indi_ in offsprings:
            indi = copy.deepcopy(indi_)
            indi_no = 'indi%02d%02d'%(self.gen_no, self.number_id)
            indi.id = indi_no
            self.number_id += 1
            indi.number_id = len(indi.units)
            self.individuals.append(indi)


    def __str__(self):
        _str = []
        for ind in self.individuals:
            _str.append(str(ind))
            _str.append('-'*100)
        return '\n'.join(_str)






def test_individual():
    params = {}
    params['min_conv'] = 30
    params['max_conv'] = 40
    params['min_pool'] = 3
    params['max_pool'] = 4
    params['max_len'] = 20
    params['image_channel'] = 3
    params['output_channel'] = [64, 128, 256, 512]
    ind = Individual(params, 0)
    ind.initialize()
    print(ind)
    print(ind.uuid())

def test_population():
    params = {}
    params['pop_size'] = 20
    params['min_conv'] = 10
    params['max_conv'] = 15
    params['min_pool'] = 3
    params['max_pool'] = 4
    params['max_len'] = 20
    params['conv_kernel'] = [1, 2, 3]
    params['conv_stride'] = [1,2, 3]
    params['pool_kernel'] = 2
    params['pool_stride'] = 2
    params['image_channel'] = 3
    params['output_channel'] = [64, 128, 256, 512]
    pop = Population(params, 0)
    pop.initialize()
    print(pop)



if __name__ == '__main__':

    test_individual()
    #test_population()




