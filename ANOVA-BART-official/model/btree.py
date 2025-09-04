import numpy as np
import pandas as pd
import copy

indicator = lambda x : np.array((x<0), dtype = np.int8)

class BTREE:
    def __init__(self, var, data, b: np.ndarray, gamma: np.ndarray, bin_function=indicator):
        self.variable = var     
        self.b = b              # (1,)
        self.gamma = gamma      # (1,)
        self.bin_function = bin_function

        x = data[:, self.variable]
        new_x = self.bin_function((x - self.b)/self.gamma) + 1e-3
        new_x1 = self.bin_function((self.b - x)/self.gamma) + 1e-3
        self.c = -np.sum(new_x1, axis = 0)/np.sum(new_x, axis = 0)
        self.c = self.c.item()

    def forward(self, data: np.ndarray):
        x = data[:, self.variable]
        new_x = self.bin_function((x - self.b)/self.gamma)
        new_x1 = self.bin_function((self.b - x)/self.gamma)
        
        result = self.c * new_x + new_x1
        # assert result.ndim == 1
        return result                               # (n,)


class BPTREE:
    def __init__(self, p: int, config: dict, structure: list[BTREE]=[], height = 0., error_variance = 1.):
        self.variables = np.arange(p)      
        self.config = config
        self.structure = structure      # list[BTREE]
        self.height = height

        if not self.structure:
            root_height = 0.0
            self.height = root_height

        self.var_error = error_variance

    def prod_structure(self, structure: list[BTREE], data: np.ndarray) -> np.ndarray:
        result = np.ones((data.shape[0],))
        if structure:
            for btree in structure:
                tmp_result = btree.forward(data)            # (n,)
                result = np.column_stack((result, tmp_result))
            result = np.prod(result, axis = 1)
        else :
            result = np.zeros((data.shape[0],))
        return result

    def new_btree(self, data: np.ndarray) -> BTREE:
        new_variable = np.random.choice(self.variables)
        if self.config['use_candidate_split_val']:
            cand = np.unique(data[:, new_variable])
            if len(cand)==1:
                raise Exception(f'Cannot use the {new_variable}th variable since it has an unique value.')
            new_b = np.random.choice(cand)
            new_b = np.array([new_b])
        else :
            new_b = np.random.uniform(self.config['b_hyperparam1'], self.config['b_hyperparam2'], size = (1,))
        new_gamma = np.random.gamma(shape = self.config['gamma_shape'], scale = self.config['gamma_scale'], size = (1,))
        new_BTREE = BTREE(new_variable, data, new_b, new_gamma)             
        return new_BTREE          

    # tree structure sampling
    def grow(self, residual: np.ndarray, data: np.ndarray):
        
        
        new_BTREE = self.new_btree(data)
        
        new_structure = copy.deepcopy(self.structure)           # list[BTREE]
        new_structure.append(new_BTREE)
        #print(data.shape)
        if len(new_structure) == data.shape[1] :
            return False
        
        assert len(new_structure) == len(self.structure) + 1

        # acceptance rate
        old_ps = self.prod_structure(self.structure, data)      # np.ndarray (n,)
        new_ps = self.prod_structure(new_structure, data)       # np.ndarray (n,)

        d_grow = len(new_structure)
        
        term1 = self.config['alpha'] * (1-self.config['alpha']*( ((1+d_grow) ** (-self.config['beta']))) ) / (d_grow**self.config['beta'] - self.config['alpha'])          
        
        term2_1 = self.var_error + self.config['var_height'] * np.sum(old_ps**2).item()
        term2_2 = self.var_error + self.config['var_height'] * np.sum(new_ps**2).item()
        term2 = np.sqrt(term2_1/term2_2)
             
        term3 = self.config['prune_prob'] / self.config['grow_prob'] 
        
        term4_1 = (np.sum(residual * old_ps).item()**2) * self.config['var_height'] / (2*term2_1)
        term4_2 = (np.sum(residual * new_ps).item()**2) * self.config['var_height'] / (2*term2_2)
                
        term4 = np.exp(term4_2 - term4_1)

        acceptance_rate = term1 * term2 * term3 * term4
        # print(f'acceptance ratio : {acceptance_rate}')
        dice = np.random.uniform(0, 1, size=(1,)).item()

        if dice < acceptance_rate :
            # print('GROW accepted!')
            self.variables = self.variables[self.variables != new_BTREE.variable]
            self.structure = new_structure
            return True
        else : 
            # print('GROW rejected!')
            return False

    def prune(self, residual: np.ndarray, data: np.ndarray):
        new_structure = copy.deepcopy(self.structure)
        prune_BTREE = np.random.choice(new_structure)
        new_structure.remove(prune_BTREE)
        assert len(new_structure) == len(self.structure) - 1

        # acceptance rate
        old_ps = self.prod_structure(self.structure, data)
        new_ps = self.prod_structure(new_structure, data)

        d_prune = len(new_structure)
        term1 = ((1+d_prune)**self.config['beta'] - self.config['alpha'])/(self.config['alpha'] * (1-self.config['alpha'] * ((2+d_prune) ** (-self.config['beta']))))
        term2_1 = self.var_error + self.config['var_height'] * np.sum(old_ps**2).item()
        term2_2 = self.var_error + self.config['var_height'] * np.sum(new_ps**2).item()
        term2 = np.sqrt(term2_1/term2_2)
        term3 = self.config['grow_prob']/self.config['prune_prob']
        
        term4_1 = np.sum(residual * old_ps).item()**2 * self.config['var_height'] / (2*term2_1)
        term4_2 = np.sum(residual * new_ps).item()**2 * self.config['var_height'] / (2*term2_2)
        
        term4 = np.exp(term4_2 - term4_1)

        acceptance_rate = term1 * term2 * term3 * term4
        dice = np.random.uniform(0, 1, size = (1,)).item()

        if dice < acceptance_rate :
            self.variables = np.append(self.variables, prune_BTREE.variable)
            self.structure = new_structure
            return True
        else :
            return False

    def change(self, residual: np.ndarray, data: np.ndarray):
        new_structure = copy.deepcopy(self.structure)
        remove_BTREE = np.random.choice(new_structure)      
        new_structure.remove(remove_BTREE)
        new_BTREE = self.new_btree(data)                    
        new_structure.append(new_BTREE)
        assert len(new_structure) == len(self.structure)

        # acceptance rate
        old_ps = self.prod_structure(self.structure, data)
        new_ps = self.prod_structure(new_structure, data)

        d_now = len(new_structure)
        term1_1 = self.var_error + self.config['var_height'] * np.sum(old_ps**2).item()
        term1_2 = self.var_error + self.config['var_height'] * np.sum(new_ps**2).item()
        term1 = np.sqrt(term1_1/term1_2)
        term2_1 = np.sum(residual * old_ps).item()**2 * self.config['var_height'] / (2*term1_1)     
        term2_2 = np.sum(residual * new_ps).item()**2 * self.config['var_height'] / (2*term1_2)     
        term2 = np.exp(term2_2 - term2_1)
        
        acceptance_rate = term1 * term2
        dice = np.random.uniform(0, 1, size = (1,)).item()

        if dice < acceptance_rate :
            self.variables = self.variables[self.variables != new_BTREE.variable]
            self.variables = np.append(self.variables, remove_BTREE.variable)
            self.structure = new_structure
            return True
        else :
            return False

    # tree height sampling
    def sample_height(self, residual: np.ndarray, data: np.ndarray):
        if self.structure:
            ps = self.prod_structure(self.structure, data)
            denom = self.var_error + self.config['var_height'] * np.sum(ps**2).item()
            mu = np.sum(residual * ps).item() * self.config['var_height'] / denom
            var = self.var_error * self.config['var_height'] / denom
            eps = np.random.normal(0, 1, size = (1,)).item()

            self.height = mu + eps * np.sqrt(var)
        else :
            self.height = 0
    
    def forward(self, data: np.ndarray):
        result = self.prod_structure(self.structure, data) * self.height    # (n,)
        return result
    

    def init_from_prior(self, data: np.ndarray):
        assert len(self.structure)==0
        # select depth
        depth_prior = []
        for d in range(self.config['max_depth']):
            d_prob = self.config['alpha'] * (1+d)**(-self.config['beta'])
            depth_prior.append(d_prob)
        depth_prior = np.array(depth_prior)
        depth_prior /= np.sum(depth_prior)
        selected_depth = np.random.choice(np.arange(1, 1+len(depth_prior)), size = 1, p = depth_prior).item()       # int type

        # sample # (selected_depth) btrees
        structure = []
        for i in range(selected_depth):
            i_btree = self.new_btree(data)
            self.variables = self.variables[self.variables != i_btree.variable]
            structure.append(i_btree)
        self.structure = structure

        # sample height
        height = np.random.normal(0, 1, size = 1)
        height *= np.sqrt(self.config['var_height'])
        self.height = height
