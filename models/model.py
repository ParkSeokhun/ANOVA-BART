from .btree import ATREE, BTREE
import numpy as np
import copy
from scipy.stats import invgamma

EPS = 1e-5


class BTREEs:
    def __init__(self, data: np.ndarray, config: dict) -> list[BTREE]:
        self.p = data.shape[1]
        self.alpha = config['alpha']
        self.beta = config['beta']
        self.max_depth = min(config['max_depth'], self.p)

        depth_prior = [self.alpha * (1+d)**(-self.beta) for d in range(self.max_depth + 1)]
        depth_prior = np.array(depth_prior)
        depth_prior = np.cumprod(depth_prior[:-1]) * (1-depth_prior[1:])
        depth_prior /= np.sum(depth_prior).item()               # 1~self.max_depth prior
        self.depth_prior = depth_prior

        w = config['feature_weight']
        w /= np.sum(w).item()
        self.w = w
        assert len(self.w) == self.p

        self.model = []
        self.variable_sets = np.array([])           # np.array of set
        for t in range(config['T_max']):
            if config['init_from_prior']:           # usually false
                tree_t = self.tree_from_prior(data, config)
            else :
                tree_t = BTREE(self.p, config['y_dist'])
            self.model.append(tree_t)
            self.variable_sets = np.append(self.variable_sets, set([a.variable for a in tree_t.structure]))
        assert len(self.model) == config['T_max']
        assert len(self.variable_sets) == config['T_max']

        self.zs = np.ones((config['T_max'],))

        self.nui = config['nui']
        if config['y_dist'] == 'normal':
            assert self.nui
        else :
            self.nui = np.random.normal(0, 1, size=1).item() * np.sqrt(config['const_var'])

    def tree_from_prior(self, data: np.ndarray, config: dict, variable_set: np.ndarray=None) -> BTREE:


        tree = BTREE(self.p, config['y_dist'])
        if variable_set is None or len(variable_set) == 0:
            selected_depth = np.random.choice(np.arange(1 + self.max_depth), size = 1, p = self.depth_prior).item()
            variable_set = np.random.choice(np.arange(self.p), size = (selected_depth,), replace = False)

        for new_variable in variable_set:
            cand = np.sort( np.unique(data[:, new_variable]))[:-1]
            
            if len(cand) > 1:
                midpoints = (cand[:-1] + cand[1:]) / 2

                #Choose one randomly
                new_b = np.random.choice(midpoints)  
            else:
                new_b = np.random.choice(cand)
         
              
            component = ATREE(new_variable, data, new_b)
            tree.structure.append(component)
            tree.variables = tree.variables[tree.variables != new_variable]
            assert len(tree.variables) + len(tree.structure) == self.p

        prior_height = np.random.normal(0, 1, size = 1).item() * np.sqrt(config['var_height']/config['T_max'])
        tree.height = prior_height
        
        return tree

    def new_atree(self, tree: BTREE, data: np.ndarray, config: dict):
        """
        RETURNs an Atree object
        This object uses a variable from the complement of variables used in the given tree
        """
        cand_weight = self.w[tree.variables]
        cand_weight /= np.sum(cand_weight).item()
        new_variable = np.random.choice(tree.variables, size = 1, p = cand_weight).item()
        
        #For split value
        cand = np.sort( np.unique(data[:, new_variable]))[:-1]
        
        n = len(cand)

        # Ratio
        if config["split_value_q"] == 1.0:
            k = n
        else:
            k = int(n * config["split_value_q"]) +1  #celing

        # quantile 
        qs = np.linspace(0, 1, k, endpoint=True)  
        new_cand = np.quantile(cand, qs)
        
        new_b = np.random.choice(new_cand)
        
 
        component = ATREE(new_variable, data, new_b)

        return component

    def grow(self, t: int, lambda_t: np.ndarray, data: np.ndarray, y: np.ndarray, config: dict):
        # new atree
        tree = copy.deepcopy(self.model[t])

        if len(tree.structure) >= config['max_depth']:
            self.model[t] = tree
            return False
        
        if len(tree.variables) == 0 :
            self.model[t] = tree
            return False

        # new component - which is not used in tree
        component = self.new_atree(tree, data, config)

        # new tree
        new_tree = copy.deepcopy(tree)
        new_tree.structure.append(component)
        assert len(new_tree.structure) == len(tree.structure) + 1
        new_tree.variables = new_tree.variables[new_tree.variables != component.variable]
        assert len(new_tree.variables) + len(new_tree.structure) == self.p

        ########### log acceptance ratio ###########
        
        #log likelihood ratio
        log_lr = new_tree.log_likelihood(lambda_t, data, y, self.nui) - tree.log_likelihood(lambda_t, data, y, self.nui)      # float
        d = len(new_tree.structure)
        
        #log prior ratio
        log_structure_pr = np.log(self.alpha) -self.beta * np.log(d) + np.log(1-self.alpha*(1+d)**(-self.beta))
        log_structure_pr += -np.log(1-self.alpha*(d**(-self.beta))) 
        log_structure_pr += -np.log( self.p - d + 1 ) 
         
        # log proposal ratio
        log_propose_ratio = np.log(config['prune_prob']) - np.log(config['grow_prob'])
        log_propose_ratio += -np.log(d)   
        log_propose_ratio += np.log( np.sum(self.w[tree.variables]) ) - np.log(self.w[component.variable]) 
        
        
        #FIXME: 1002
        #For fast MCMC we did not calculate below.
        # log_structure_pr += - np.log( len(  np.unique(data[:, component.variable]) ) )
        # cand = np.sort( np.unique(data[:, component.variable]) )[:-1]
        # n = len(cand)
        # k = int(n * config["split_value_q"]) +1  #celing
        # log_propose_ratio += np.log( k )
        
                
        log_acceptance_ratio = log_lr + log_structure_pr + log_propose_ratio
        if log_acceptance_ratio > 0 :
            acceptance_ratio = 1.
        else :
            acceptance_ratio = np.exp(log_acceptance_ratio).item()

        # dice
        dice = np.random.uniform(0, 1, size = 1).item()
        if dice < acceptance_ratio :
            self.model[t] = new_tree
            self.variable_sets[t] = set([a.variable for a in new_tree.structure])
            return True
        else :
            self.model[t] = tree
            return False

    def prune(self, t: int, lambda_t: np.ndarray, data: np.ndarray, y: np.ndarray, config: dict):
        tree = copy.deepcopy(self.model[t])
        d = len(tree.structure)

        if len(tree.structure) == 0 :
            self.model[t] = tree
            return False

        # select atree
        what_to_remove = np.random.choice(np.arange(d), size = 1).item()

        # new tree
        new_tree = copy.deepcopy(tree)
        removed_atree = new_tree.structure.pop(what_to_remove)
        assert len(new_tree.structure) == len(tree.structure) - 1
        new_tree.variables = np.append(new_tree.variables, removed_atree.variable)
        assert len(new_tree.variables) + len(new_tree.structure) == self.p

        # log acceptance ratio
        
        # Likelihood ratio
        log_lr = new_tree.log_likelihood(lambda_t, data, y, self.nui) - tree.log_likelihood(lambda_t, data, y, self.nui)      # float
        
        # prior ratio
        log_structure_pr = np.log(1-self.alpha*(d**(-self.beta)))    
        log_structure_pr += -np.log(self.alpha) + self.beta * np.log(d) 
        log_structure_pr += -np.log(1-self.alpha*((1+d)**(-self.beta)))
        log_structure_pr +=  np.log( self.p - d + 1 )  
             
        
        # proposal ratio
        log_propose_ratio = np.log(config['grow_prob']) - np.log(config['prune_prob'])
        log_propose_ratio += np.log( d )
        log_propose_ratio += np.log(self.w[removed_atree.variable]) - np.log( np.sum(self.w[new_tree.variables]) ) 
        


        log_acceptance_ratio = log_lr + log_structure_pr + log_propose_ratio

        acceptance_ratio = np.exp(log_acceptance_ratio).item()

        # dice
        dice = np.random.uniform(0, 1, size = 1).item()
        if dice < acceptance_ratio :
            self.model[t] = new_tree
            self.variable_sets[t] = set([a.variable for a in new_tree.structure])
            return True
        else :
            self.model[t] = tree
            return False

    def change(self, t: int, lambda_t: np.ndarray, data: np.ndarray, y: np.ndarray, config: dict):
        tree = copy.deepcopy(self.model[t])
        d = len(tree.structure)

        if d == 0:
            return False
        
        if len(tree.variables) == 0 :
            self.model[t] = tree
            return False

        # new component
        component = self.new_atree(tree, data, config)

        # select atree to remove
        what_to_remove = np.random.choice(np.arange(d), size = 1).item()        # idx

        # new tree
        new_tree = copy.deepcopy(tree)
        

        
        ## remove
        removed_atree = new_tree.structure.pop(what_to_remove)
        new_tree.variables = np.append(new_tree.variables, removed_atree.variable)
        ## append
        new_tree.structure.append(component)
        new_tree.variables = new_tree.variables[new_tree.variables != component.variable]
        assert len(new_tree.structure) == len(tree.structure)
        assert len(new_tree.variables) + len(new_tree.structure) == self.p

        # log acceptance ratio
        
        #likelihood ratio
        log_lr = new_tree.log_likelihood(lambda_t, data, y, self.nui) - tree.log_likelihood(lambda_t, data, y, self.nui)
        
        #proposal ratio
        log_propose_ratio = np.log(self.w[removed_atree.variable]) + np.log(np.sum(self.w[tree.variables])) 
        log_propose_ratio += - np.log(self.w[component.variable]) - np.log(np.sum(self.w[new_tree.variables]))
        #log_propose_ratio += np.log( deleted_Aj / new_Aj )
        
        #log_propose_ratio = 0.0
        
        log_acceptance_ratio = log_lr + log_propose_ratio
        acceptance_ratio = np.exp(log_acceptance_ratio).item()

        # dice
        dice = np.random.uniform(0, 1, size = 1).item()
        if dice < acceptance_ratio :
            self.model[t] = new_tree
            self.variable_sets[t] = set([a.variable for a in new_tree.structure])
            return True
        else :
            self.model[t] = tree
            return False


    def b_update(self, t: int, lambda_t: np.ndarray, data: np.ndarray, y: np.ndarray, config: dict):
        tree = copy.deepcopy(self.model[t])
        new_tree = copy.deepcopy(self.model[t])
        prop_ratios = []
        
        for idx, atree in enumerate(tree.structure):
            b_cand_sorted = np.sort(np.unique(data[:, atree.variable]))[:-1]

            cur_b = atree.b
            ll_cur = tree.log_likelihood(lambda_t, data, y, self.nui)
            l_cur = np.exp(ll_cur).item()
            cur_b_idx = np.searchsorted(b_cand_sorted, cur_b)

            if cur_b in b_cand_sorted:
                right_b_idx = cur_b_idx + 1
            else :
                right_b_idx = cur_b_idx
            left_b_idx = cur_b_idx - 1

            # propose new split value
            if right_b_idx < len(b_cand_sorted) :
                b_right = b_cand_sorted[right_b_idx]
                right_tree = copy.deepcopy(tree)
                right_atree = ATREE(atree.variable, data, b_right)

                right_tree.structure[idx] = right_atree
                ll_right = right_tree.log_likelihood(lambda_t, data, y, self.nui)
                l_right = np.exp(ll_right).item()
                del right_tree
            else :
                right_atree = None
                l_right = -EPS

            if left_b_idx >= 0 :
                b_left = b_cand_sorted[left_b_idx]
                left_tree = copy.deepcopy(tree)
                left_atree = ATREE(atree.variable, data, b_left)

                left_tree.structure[idx] = left_atree
                ll_left = left_tree.log_likelihood(lambda_t, data, y, self.nui)
                l_left = np.exp(ll_left).item()
                del left_tree
            else :
                left_atree = None
                l_left = -EPS

            l_list = [l_left, l_cur, l_right]
            cand = [left_atree, atree, right_atree]

            old_cand_prob = np.array(l_list)+EPS
            old_cand_prob /= np.sum(old_cand_prob).item()
            new_idx = np.random.choice(np.arange(len(cand)), size = 1, p = old_cand_prob).item()

            new_tree.structure[idx] = cand[new_idx]
            assert cand[new_idx]

            if new_idx == 2:
                out_b_idx = right_b_idx + 1
                if out_b_idx < len(b_cand_sorted):
                    b_out = b_cand_sorted[out_b_idx]
                    out_tree = copy.deepcopy(tree)
                    out_atree = ATREE(atree.variable, data, b_out)
                    out_tree.structure[idx] = out_atree
                    ll_out = out_tree.log_likelihood(lambda_t, data, y, self.nui)
                    l_out = np.exp(ll_out).item()
                    del out_tree
                else :
                    l_out = -EPS
                new_l_list = l_list + [l_out]
                new_l_list = np.array(new_l_list)+EPS               # (l-1, l0, l1, l2)
                prop_ratio = (new_l_list[1] * np.sum(new_l_list[:-1]))/(new_l_list[2] * np.sum(new_l_list[1:]) + EPS)

            elif new_idx == 0:
                out_b_idx = left_b_idx - 1
                if out_b_idx >= 0:
                    b_out = b_cand_sorted[out_b_idx]
                    out_tree = copy.deepcopy(tree)
                    out_atree = ATREE(atree.variable, data, b_out)
                    out_tree.structure[idx] = out_atree
                    ll_out = out_tree.log_likelihood(lambda_t, data, y, self.nui)
                    l_out = np.exp(ll_out).item()
                    del out_tree
                else :
                    l_out = -EPS
                new_l_list = [l_out] + l_list
                new_l_list = np.array(new_l_list)+EPS               # (l-2, l-1, l0, l1)
                prop_ratio = (new_l_list[2] * np.sum(new_l_list[1:]))/(new_l_list[1]*np.sum(new_l_list[:-1]) + EPS)
            
            else :
                prop_ratio = np.array([1.])
            
            prop_ratios.append(prop_ratio.item())

        log_lr = new_tree.log_likelihood(lambda_t, data, y, self.nui) - tree.log_likelihood(lambda_t, data, y, self.nui)
        log_pr = np.sum(np.log(np.array(prop_ratios))).item()

        log_acceptance_ratio = log_lr + log_pr
        log_dice = np.log(np.random.uniform(0, 1, size = 1)).item()
        
        if log_dice < log_acceptance_ratio:
            self.model[t] = new_tree
            return True
        else :
            return False


    def height_update(self, t: int, lambda_t: np.ndarray, data: np.ndarray, y: np.ndarray, config: dict):
        tree = copy.deepcopy(self.model[t])

        # L-step Leapfrog
        momentum = np.random.normal(0, 1, size = 1).item()
        for l in range(config['leapfrog_L']):
            if l == 0:
                new_tree = copy.deepcopy(tree)
                stein_score = -new_tree.height * config['T_max']/config['var_height'] + new_tree.height_score_function(lambda_t, data, y, self.nui)
                new_momentum = momentum + (config['step_size']/config['leapfrog_L']) * stein_score / 2


            new_height = new_tree.height + (config['step_size']/config['leapfrog_L']) * new_momentum        # 250710
            new_tree.height = new_height
            stein_score = -new_tree.height * config['T_max']/config['var_height'] + new_tree.height_score_function(lambda_t, data, y, self.nui)
            new_momentum += (config['step_size']/config['leapfrog_L']) * stein_score
        new_momentum -= (config['step_size']/config['leapfrog_L']) * stein_score/2

        log_acceptance_ratio = new_tree.log_likelihood(lambda_t, data, y, self.nui) - tree.log_likelihood(lambda_t, data, y, self.nui)          # log LR
        log_acceptance_ratio -= (new_tree.height**2 - tree.height**2) * config['T_max']/(2*config['var_height'])                                # log prior ratio
        
        # try :
        #     log_acceptance_ratio -= (new_momentum**2 - momentum**2)/2
        # except :
        #     momentum_diff = new_momentum - momentum
        #     log_acceptance_ratio -= np.sum(2 * momentum * momentum_diff + momentum_diff**2)/2
            
        log_acceptance_ratio -= (new_momentum**2 - momentum**2)/2
        acceptance_ratio = np.exp(log_acceptance_ratio).item()

        # dice
        dice = np.random.uniform(0, 1, size = 1).item()
        if dice < acceptance_ratio :
            self.model[t] = new_tree
            return True
        else :
            self.model[t] = tree
            return False

    def btrees_update(self, data: np.ndarray, y: np.ndarray, config: dict):
        lambda_t = self.forward(data, config)
        gpc_prob = np.array([config['grow_prob'], config['prune_prob'], config['change_prob']]); gpc_prob_sum = np.sum(gpc_prob).item()
        gpc_prob /= gpc_prob_sum

        self.gpc_log = ''
        self.b_log = ''
        self.height_log = ''

        for t in range(config['T_max']):
            # S_t
            if self.model[t].z :
                lambda_t -= self.model[t].forward(data)

                gpc = np.random.choice(['grow', 'prune', 'change'], size = 1, p = gpc_prob).item()
                if gpc == 'grow':
                    updated = self.grow(t, lambda_t, data, y, config)
                elif gpc == 'prune':
                    updated = self.prune(t, lambda_t, data, y, config)
                else :
                    updated = self.change(t, lambda_t, data, y, config)

                if updated :
                    self.gpc_log += gpc[0]
                else :
                    self.gpc_log += 's'

                # b
                if config["b_update"] == True:
                    if self.model[t].structure:
                        b_updated = self.b_update(t, lambda_t, data, y, config)
                        if b_updated:
                            self.b_log += 'Y'
                        else :
                            self.b_log += '_'
                    else :
                        self.b_log += 'Z'

                # beta
                height_updated = self.height_update(t, lambda_t, data, y, config)   # 250608
                if height_updated : 
                    self.height_log += 'H'
                else : 
                    self.height_log += '_'
                # print(self.height_log)

                lambda_t += self.model[t].forward(data)

            else :
                old_variable = self.variable_sets[t]
                k = int(np.sum(self.zs).item())

                # propose new variable set
                from_prior = np.random.uniform(0, 1, size=1).item()
                if from_prior < config['M'] / (config['M'] + k):
                    ref_depth = np.random.choice(np.arange(self.max_depth)+1, size = 1, p = self.depth_prior).item()        # must be int
                    ref_variable = np.random.choice(np.arange(self.p), size = (ref_depth,), replace = False)                # choose variable set
                    ref_variable = set(ref_variable)
                else :
                    ref_tree_idx = np.random.choice(np.arange(k), size = 1).item()
                    ref_variable = self.variable_sets[np.where(self.zs == 1)][ref_tree_idx]     # set
                    assert isinstance(ref_variable, set) or len(ref_variable) == 0
                    variable_cand = np.array(list(set(range(self.p)) - ref_variable))       # array(int)
                    if (len(variable_cand) > 0) and (len(ref_variable) < config['max_depth']):
                        cand_weight = self.w[variable_cand]
                        cand_weight /= np.sum(cand_weight).item()
                        var_to_add = np.random.choice(variable_cand, size = 1, p = cand_weight).item()      # int
                        ref_variable.add(var_to_add)        # set
                    assert len(ref_variable) >= 1
                
                # cal acceptance ratio
                if len(old_variable) == 0:
                    log_acceptance_ratio = 0.
                else :
                    lx1 = np.log(self.weight_pot_ref_tree(old_variable)).item() \
                        - np.log(self.depth_prior[len(old_variable)-1]).item() + self.log_combination(self.p, len(old_variable)) - np.log(config['M']).item()
                    lx2 = np.log(self.weight_pot_ref_tree(ref_variable)).item() \
                        - np.log(self.depth_prior[len(ref_variable)-1]).item() + self.log_combination(self.p, len(ref_variable)) - np.log(config['M']).item()
                    log_acceptance_ratio = np.logaddexp(0.0, lx1) - np.logaddexp(0.0, lx2)

                # update
                dice = np.random.uniform(0, 1)
                log_dice = np.log(dice).item()
                if log_dice < log_acceptance_ratio :
                    new_tree = self.tree_from_prior(data, config, np.array(list(ref_variable)))
                    new_tree.z = 0
                    self.model[t] = new_tree
                    self.variable_sets[t] = ref_variable
                    self.gpc_log += 'z'
                    self.b_log += '_'
                    self.height_log += '_'
                else :
                    self.gpc_log += 's'
                    self.b_log += '_'
                    self.height_log += '_'
                
        self.w_update(config)
            
    #FIXME:0928 
    def w_update(self,config: dict):
        new_w = np.zeros(self.p) + 1e-10
        full_idx = np.arange(1, self.p)  
        for t in range(0,config["T_max"]):
            if self.model[t].z:
                missing = np.setdiff1d(full_idx, self.model[t].variables)
                new_w[missing] += 1
        self.w = new_w
                  
                  
    def z_update(self, data: np.ndarray, y: np.ndarray, config: dict):
        z_idx = np.random.choice(np.arange(config['T_max']), size = 1).item()

        z_tree = copy.deepcopy(self.model[z_idx])
        assert z_tree.z == self.zs[z_idx].item()
        new_z_tree = copy.deepcopy(z_tree); new_z_tree.z = 1 - z_tree.z

        lambda_z_idx = self.forward(data, config); z_tree_forward = z_tree.forward(data)
        lambda_z_idx -= z_tree_forward

        z_tree_log_likelihood = z_tree.log_likelihood(lambda_z_idx, data, y, self.nui)
        new_z_tree_log_likelihood = new_z_tree.log_likelihood(lambda_z_idx, data, y, self.nui)
        log_lr = new_z_tree_log_likelihood - z_tree_log_likelihood

        n = data.shape[0]
        cur_zs_sum = np.sum(self.zs).item()
        if z_tree.z :
            log_pr = config['c_star'] * np.log(n).item() + np.log(config['T_max']-cur_zs_sum+1).item() - np.log(cur_zs_sum).item()
        else :
            log_pr = -config['c_star'] * np.log(n).item() + np.log(cur_zs_sum + 1).item() - np.log(config['T_max']-cur_zs_sum-1+1).item()

        log_acceptance_ratio = log_lr + log_pr
        acceptance_ratio = np.exp(log_acceptance_ratio).item()

        # dice
        dice = np.random.uniform(0, 1, size = 1).item()
        if dice < acceptance_ratio:
            self.model[z_idx] = new_z_tree
            self.zs[z_idx] = 1 - self.zs[z_idx]
            assert self.model[z_idx].z == self.zs[z_idx].item()
            self.z_updated = True
            # print(f'z updated from {1-new_z_tree.z} to {new_z_tree.z}')
        else :
            self.model[z_idx] = z_tree
            self.z_updated = False
            # print(f'z update rejected.')

            

    def nui_update(self, data: np.ndarray, y: np.ndarray, config: dict):
        if config['y_dist'] == 'normal':
            assert self.nui['sigma2']
            forwarded_value = self.forward(data, config)
            new_inv_gamma_1 = (self.nui['inv_gamma_nu'] + data.shape[0])/2
            new_inv_gamma_2 = (self.nui['inv_gamma_nu'] * self.nui['inv_gamma_lambda'] + np.sum((y - forwarded_value)**2).item())/2   
            self.nui['sigma2'] = invgamma.rvs(new_inv_gamma_1,0,new_inv_gamma_2, size=1).item()

        elif config['y_dist'] == 'ber':
            assert self.nui
            assert config['const_var'] > 0
            assert config['const_step_size'] > 0
            forwarded_value = self.forward(data, config)
            exp_forwarded_value = np.exp(forwarded_value)
            momentum = np.random.normal(0, 1, size = 1).item()
            # JUST USE LANGEVIN FROM NOW ON
            log_posterior = np.sum(forwarded_value * y - np.log(1+exp_forwarded_value)).item() - self.nui**2/(2*config['const_var'])     # -U(beta_0)
            log_posterior_diff = np.sum(y - exp_forwarded_value/(1+exp_forwarded_value)).item() - self.nui / config['const_var']        # -dU(beta_0)/d(beta_0)
            new_nui = self.nui + (config['const_step_size'] ** 2 * log_posterior_diff)/2 + config['const_step_size'] * momentum

            # acceptance ratio
            new_forwarded_value = forwarded_value - self.nui + new_nui
            exp_new_forwarded_value = np.exp(new_forwarded_value)
            new_log_posterior = np.sum(new_forwarded_value * y - np.log(1+exp_new_forwarded_value)).item() - new_nui**2/(2*config['const_var'])     # -U(beta_0*)
            new_log_posterior_diff = np.sum(y - exp_new_forwarded_value/(1+exp_new_forwarded_value)).item() - new_nui / config['const_var']        # -dU(beta_0*)/d(beta_0*)
            new_momentum = momentum + config['const_step_size'] * log_posterior_diff / 2 + config['const_step_size'] * new_log_posterior_diff / 2

            log_acceptance_ratio = (new_log_posterior - log_posterior) - (new_momentum**2 - momentum**2)/2

            dice = np.random.uniform(0, 1, size = 1).item()
            if np.log(dice) < log_acceptance_ratio : 
                self.nui = new_nui
        
        elif config['y_dist'] == 'poisson':
            assert self.nui
            assert config['const_var'] > 0
            assert config['const_step_size'] > 0
            forwarded_value = self.forward(data, config)
            momentum = np.random.normal(0, 1, size = 1).item()
            log_posterior = np.sum(y * forwarded_value - np.exp(forwarded_value)).item() - self.nui**2/(2*config['const_var'])
            log_posterior_diff = np.sum(y - np.exp(forwarded_value)).item() - self.nui / config['const_var']
            new_nui = self.nui + (config['const_step_size'] ** 2 * log_posterior_diff)/2 + config['const_step_size'] * momentum

            new_forwarded_value = forwarded_value - self.nui + new_nui
            new_log_posterior = np.sum(y * new_forwarded_value - np.exp(new_forwarded_value)).item() - new_nui**2/(2*config['const_var'])
            new_log_posterior_diff = np.sum(y - np.exp(new_forwarded_value)).item() - new_nui / config['const_var']
            new_momentum = momentum + config['const_step_size'] * log_posterior_diff / 2 + config['const_step_size'] * new_log_posterior_diff / 2

            log_acceptance_ratio = (new_log_posterior - log_posterior) - (new_momentum**2 - momentum**2) / 2

            dice = np.random.uniform(0, 1, size = 1).item()
            if np.log(dice) < log_acceptance_ratio :
                self.nui = new_nui


    def forward(self, data: np.ndarray, config: dict) -> np.ndarray:
        result = np.zeros((data.shape[0],))
        for tree_t in self.model:
            t_forward = tree_t.forward(data) 
            result += t_forward
        if config['y_dist'] != 'normal':
            result += self.nui
        return result
    

    def evaluate(self, ss: int, data: np.ndarray, y: np.ndarray, test_data: np.ndarray, test_y: np.ndarray, config, print_res = False):
        assert data.shape[1] == test_data.shape[1]
        assert data.shape[0] == y.shape[0]
        assert test_data.shape[0] == test_y.shape[0]

        if config['y_dist'] == 'normal':
            train_fitted_value = self.forward(data, config)
            train_rmse = np.sqrt(np.mean((y - train_fitted_value)**2)).item() * config['y_std']
            
            test_fitted_value = self.forward(test_data, config)
            test_rmse = np.sqrt(np.mean((test_y - test_fitted_value)**2)).item() * config['y_std']

            self.train_metric = np.round(train_rmse, 3)
            self.test_metric = np.round(test_rmse, 3)
        
        elif config['y_dist'] == 'ber':
            train_fitted_value = self.forward(data, config)
            from sklearn.metrics import roc_auc_score
            train_auroc = roc_auc_score(y, train_fitted_value)

            test_fitted_value = self.forward(test_data, config)
            test_auroc = roc_auc_score(test_y, test_fitted_value)

            self.train_metric = np.round(train_auroc, 4)
            self.test_metric = np.round(test_auroc, 4)

        elif config['y_dist'] == 'poisson':
            train_fitted_value = self.forward(data, config)
            train_fitted_value = np.exp(train_fitted_value)
            train_rmse = np.sqrt(np.mean((y - train_fitted_value)**2)).item()
            
            test_fitted_value = self.forward(test_data, config)
            test_fitted_value = np.exp(test_fitted_value)
            test_rmse = np.sqrt(np.mean((test_y - test_fitted_value)**2)).item()

            self.train_metric = np.round(train_rmse, 3)
            self.test_metric = np.round(test_rmse, 3)

        if print_res :
            print(f'fold {config["fold"]} {ss+1}th Trees \ttrain : {self.train_metric}\ttest : {self.test_metric}\t # of Tree :{np.sum(self.zs).astype(int).item()}')

    def log_combination(self, n: int, x:int):
        assert n >= x
        res = 0.
        for i in range(x):
            res += np.log(n-i).item() - np.log(x-i).item()
        return res
    
    def weight_pot_ref_tree(self, variable: set):
        """
        variable : S_t or S_t^new
        sum_{i:z_i=1} I(\exists j_i\in S_i^c s.t. S_i \cup {j_i} = S_t) * (w_{j_i})/(\sum_{v\in S_i^c} w_v)
        """
        weight = 0.
        z_one_variable_sets = self.variable_sets[np.where(self.zs == 1)]
        for tmp_variable_set in z_one_variable_sets:
            # tmp_variable_set : S_i, variable : S_t or S_t^new
            if len(tmp_variable_set - variable) == 0 and len(variable - tmp_variable_set) == 1:
                tmp_variable = int(list(variable - tmp_variable_set)[0])        # j_i
                assert isinstance(tmp_variable, int)
                tmp_variable_weight = self.w[tmp_variable].item()
                tmp_compliment_variable_weight = np.sum(self.w).item() - np.sum( self.w[list(tmp_variable_set)] ).item()
                weight += tmp_variable_weight/tmp_compliment_variable_weight
        
        return weight

        

