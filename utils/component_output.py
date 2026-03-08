import numpy as np



def component_output_(samples,st_idx,ed_idx,test_x,max_order,config):
    k_max = config["T_max"]
    
    component_list = []
    
    for z in range(st_idx,ed_idx):
        local_models = samples[z].model
        
        
        for k in range(0,k_max):
            
            component_length = len(local_models[k].structure)
            local_component_list = []
            for j in range(0,component_length):
                local_component_list.append(local_models[k].structure[j].variable)
                #local_component_list.append(local_models[k].structure[j].variable)
            component_list.append(local_component_list)


    #unique_component_list = list(set(tuple(x) for x in component_list if x))  # 빈 리스트 제외
    
    unique_component_list = sorted(
        {tuple(sorted(x)) for x in component_list if x and len(x) <= max_order},  # 길이 max_order 미만
        key=lambda t: (len(t), t)
    )
        
    
    output_forcomp_list = np.zeros([test_x.shape[0],len(unique_component_list)])

    for i in range(0,len(unique_component_list)):
        
        local_component = unique_component_list[i]
        
        for z in range(st_idx,ed_idx):
            local_models = samples[z].model

            for k in range(0,k_max):
                local_btree_component_list = []
                component_length = len(local_models[k].structure)
                for j in range(0,component_length):
                    local_btree_component_list.append(local_models[k].structure[j].variable)

                if set(local_btree_component_list) == set(local_component):
                    
                    local_output = local_models[k].forward(test_x)
                    
                    output_forcomp_list[:,i] += local_output /(ed_idx-st_idx)
                    
                
    return unique_component_list, output_forcomp_list
