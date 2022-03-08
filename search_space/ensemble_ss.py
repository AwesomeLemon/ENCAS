import itertools

class EnsembleSearchSpace:
    def __init__(self, ss_names_list, ss_kwargs_list):
        from search_space import make_search_space
        self.search_spaces = [make_search_space(ss_name, **ss_kwargs)
                    for ss_name, ss_kwargs in zip(ss_names_list, ss_kwargs_list)]
        self.n_ss = len(self.search_spaces)

    def sample(self, n_samples=1):
        return list(zip([ss.sample(n_samples) for ss in self.search_spaces]))

    def initialize(self, n_doe):
        return list(zip(*[ss.initialize(n_doe) for ss in self.search_spaces]))

    def encode(self, configs, if_return_separate=False):
        # returns concatenated encoding of all the configs as a single flat list

        encoded_configs = [ss.encode(config) for ss, config in zip(self.search_spaces, configs)]
        if if_return_separate:
            return encoded_configs
        encoded = list(itertools.chain(*encoded_configs))
        return encoded

    def decode(self, enc_configs):
        # takes configs concatenated to a single string
        # returns a list of configs, each of which is a dictionary

        enc_configs_separated = []
        for ss in self.search_spaces:
            enc_configs_part = enc_configs[:ss.encoded_length]
            enc_configs = enc_configs[ss.encoded_length:]
            enc_configs_separated.append(enc_configs_part)
        decoded = [ss.decode(config) for ss, config in zip(self.search_spaces, enc_configs_separated)]
        return decoded