from pySOT.surrogate import RBFInterpolant, CubicKernel, TPSKernel, LinearTail, ConstantTail
import numpy as np

class RBF:
    """ Radial Basis Function """

    def __init__(self, kernel='cubic', tail='linear', alphabet=None, alphabet_lb=None):
        self.kernel = kernel
        self.tail = tail
        self.name = 'rbf'
        self.model = None
        self.alphabet = alphabet
        self.alphabet_lb = alphabet_lb

    def fit(self, train_data, train_label):
        if self.kernel == 'cubic':
            kernel = CubicKernel
        elif self.kernel == 'tps':
            kernel = TPSKernel
        else:
            raise NotImplementedError("unknown RBF kernel")

        if self.tail == 'linear':
            tail = LinearTail
        elif self.tail == 'constant':
            tail = ConstantTail
        else:
            raise NotImplementedError("unknown RBF tail")

        idx_unique = np.unique(train_data, axis=0, return_index=True)[1]

        self.model = RBFInterpolant(dim=train_data.shape[1], kernel=kernel(), tail=tail(train_data.shape[1]),
                                    lb=self.alphabet_lb, ub=self.alphabet)

        for i in range(len(train_data[idx_unique, :])):
            self.model.add_points(train_data[idx_unique, :][i, :], train_label[idx_unique][i])

    def predict(self, test_data):
        test_data = np.array(test_data)
        assert len(test_data.shape) == 2
        return self.model.predict(test_data)
