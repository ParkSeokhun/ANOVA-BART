import numpy as np

#sigmoid = lambda x : 1/(1+np.exp(-x))
#indicator = lambda x : np.array((x >= 0), dtype = np.int8)

def indicator(x):
    return np.array(x >=0)

def log_factorial(n: int):
    assert n >= 0
    if n == 0:
        return 0.
    
    res = 0.
    for i in range(n):
        res += np.log(n-i).item()
    return res
    
class ATREE:
    def __init__(self, var, data, b: float):
        self.variable = var     # var >0
        self.b = b                   
        self.bin_function = indicator

        x = data[:, self.variable]
        new_x = self.bin_function(x - self.b)
        new_x1 = self.bin_function(self.b - x)

        self.eta = np.mean(new_x, axis = 0).item()
        self.c = -np.sum(new_x1, axis = 0)/np.sum(new_x, axis = 0)
        self.c = self.c.item()

    def forward(self, data: np.ndarray) -> np.ndarray:
        
        x = data[:, self.variable]
        new_x = self.bin_function(x - self.b)
        new_x1 = self.bin_function(self.b - x)
        
        result = self.c * new_x + new_x1
        return result                               # (n,)

    

    ## stability score 
    def component_forward(self, var_array: np.ndarray) -> np.ndarray:
        var_array = var_array.squeeze()     # (n,)
        assert var_array.ndim == 1
        new_x = self.bin_function(var_array - self.b)
        new_x1 = self.bin_function(self.b - var_array)
        result = self.c * new_x + new_x1            # (n,)

        return result


class BTREE:
    def __init__(self, p: int, y_dist: str, structure: list[ATREE]=None, height = 0. , z: int=1):
        self.variables = np.arange(p)      # variable array
        self.y_dist = y_dist
        self.structure = structure if structure is not None else []      # list[BTREE]
        self.height = height
        self.z = z

        if not self.structure:
            root_height = 0.0
            self.height = root_height

    def prod_structure(self, data: np.ndarray) -> np.ndarray:
        result = np.ones((data.shape[0],))
        if self.structure:
            for btree in self.structure:
                tmp_result = btree.forward(data)            # (n,)
                result = np.column_stack((result, tmp_result))
            result = np.prod(result, axis = 1)
        else :
            result = np.zeros((data.shape[0],))
        return result

    def log_likelihood(self, lambda_t: np.ndarray, data: np.ndarray, y:np.ndarray, nui = None) -> float:
        """
        exact log likelihood
        """
        # lambda_t \in \mathbb{R}^n
        forward_t = lambda_t + self.height * self.z * self.prod_structure(data)
        if self.y_dist == 'normal':
            assert nui['sigma2']
            var_error = nui['sigma2']
            # y ~ N(forward_t, var_error)
            log_likelihood = -np.sum((y - forward_t)**2).item()/(2*var_error)
            log_likelihood += (-np.log(2*np.pi*var_error)/2) * y.shape[0]
        elif self.y_dist == 'poisson':
            # y ~ Poisson(exp(forward_t))
            # forward_t += nui
            log_likelihood = np.sum(y * forward_t - np.exp(forward_t)).item()
            log_likelihood += -log_factorial(y)
        elif self.y_dist == 'ber':
            # y ~ Ber(exp(forward_t)/(1+exp(forward_t)))
            # forward_t += nui
            log_likelihood = np.sum(y*forward_t - np.log(1+np.exp(forward_t))).item()

        return log_likelihood

    def height_score_function(self, lambda_t: np.ndarray, data: np.ndarray, y: np.ndarray, nui = None) -> float:
        """
        diff of log likelihood for beta (height)
        """
        spline = self.prod_structure(data)                      # (n,)
        forward_t = lambda_t + self.height * self.z * spline    # (n,)
        # print(f'y shape : {y.shape}, spline shape : {spline.shape}')
        if self.y_dist == 'normal':
            assert nui['sigma2']
            var_error = nui['sigma2']
            height_score = np.sum((y - forward_t) * spline).item()/var_error    # scalar
            height_score *= self.z
        elif self.y_dist == 'poisson':
            # forward_t += nui
            height_score = np.sum(y * spline - np.exp(forward_t) * spline).item()       # scalar
            height_score *= self.z
        elif self.y_dist == 'ber':
            # forward_t += nui
            exp_forward_t = np.exp(forward_t)               # (n,)
            height_score = np.sum(y * spline - spline * exp_forward_t/(1+exp_forward_t))
            height_score *= self.z                                              # scalar

        return height_score
    

    def forward(self, data: np.ndarray) -> np.ndarray:
        result = self.prod_structure(data) * self.height * self.z    # (n,)
        return result
