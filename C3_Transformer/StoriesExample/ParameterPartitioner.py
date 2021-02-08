import torch, torch.nn as nn, torch.optim as optim, numpy as np

def partition_params(model):
    '''
    model: pytorch model; which subclasses nn.Module
    returns set; of param names
    '''
    def params_not_in_module(model):
        children = lambda dct: set([mn.split('.')[0] for mn, m in dct])
        child_modules = children(model.named_modules())
        child_params = children(model.named_parameters())
        return list(child_params - child_modules)

    partition = set([])
    for mn, m in model.named_modules():
        for pn, p in m.named_parameters():
            partition.add('%s.%s' % (mn, pn) if mn else pn)

    for pn in params_not_in_module(model): 
        partition.add(pn)

    return partition

def rec(conditions, paths, depth=0):
    if len(conditions)<=depth: return paths
    cond = conditions[depth]
    # partition paths
    satisfies, fails = [], []
    for path in paths:
        if cond in path:
            satisfies.append(path)
        else:
            fails.append(path)
    return (rec(conditions, fails, depth+1), 
            rec(conditions, satisfies, depth+1))

def multiple_partition(model, conditions, comb=None):
    '''
    partitions all of the model parameters,
    in the order of the conditions given 
    '''
    prt = partition_params(model)# returns all parameters 
    partition = rec(conditions, prt)

    def rec_tup(ixlist, tup):
        if len(ixlist)>1:
            return rec_tup(ixlist[1:], tup[ixlist[0]])
        return tup[ixlist[0]]

    n = len(conditions)
    x = np.array([0, 1], dtype=int)
    xs = np.array(np.meshgrid(*([x]*n))).T.reshape(-1, n)
    remaining = set([''.join([str(xi) for xi in x_]) for x_ in xs])

    if comb is not None:
        p = {}    
        for pname, ixs in comb.items():
            if ixs is not None:
                p[pname] = []
                for ix in ixs:
                    p[pname] += rec_tup(ix, partition)
                    remaining.remove(''.join([str(xi) for xi in ix]))
        # put the remainder into the key with value None
        for pname, ixs in comb.items():
            if ixs is None:
                p[pname] = []
                for r in remaining:
                    ix = [int(i) for i in r]
                    p[pname] += rec_tup(ix, partition)
        return p
    return {''.join([str(xi) for xi in ix]): rec_tup(ix, partition) for ix in xs}

def get_param_dict(model, partition, kwargs):
    '''
    '''
    assert sorted(partition.keys())==sorted(kwargs.keys()), 'partition, opt_kwargs must have same keys'
    return {pn: p for pn, p in model.named_parameters()}

def optim_groups(model, partition, opt_kwargs):
    '''
    model: pytorch model; which subclasses nn.Module
    partition: dict; keys are types and values are param names
    opt_kwargs: dict; keys are types and values are optimiser kwargs
    returns list; each elem are dicts for each optimiser
    '''
    param_dict = get_param_dict(model, partition, opt_kwargs)
    optim_groups = []

    for type_, param_names in partition.items():
        group = {"params": [param_dict[pn] for pn in sorted(list(param_names))]}
        group.update(opt_kwargs[type_])
        optim_groups.append(group)

    return optim_groups

def apply_init(model, init_kwargs, partition=None, pr=False):
    '''
    two cases: (1) partition==None, else (2)
    (1): Apply different init to different modules and weight and biases.
         init_kwargs is a tuple, (weight_kwargs, bias_kwargs) this applies 
         initialisation to modules indicated by the keys of weight_kwargs 
         and keys of bias_kwargs. The values are an init fn and its argum-
         ents and named arguments.
    (2): Apply different init to different parameter names. 
         init_kwargs is a dict with keys as the param name groups and val-
         ues are init fn, its arguments and named arguments. partition is 
         a dict with keys as the param name groups and values are a list 
         of param names.
    '''

    if partition is not None:
        param_dict = get_param_dict(model, partition, init_kwargs)

        for type_, param_names in partition.items():
            init_fn, args, kwargs = init_kwargs[type_]
            for pn in sorted(list(param_names)):
                init_fn(param_dict[pn], *args, **kwargs)
                if pr: print(pn+': '+init_fn.__name__)
    else:
        w_kw, b_kw = init_kwargs
        for m in model.modules():
            key = str(m).split('(')[0] 
            srg = key

            if (key in w_kw or 'Default' in w_kw) and hasattr(m, 'weight') and m.weight is not None:
                init_fn, args, kwargs = w_kw[key] if key in w_kw else w_kw['Default']
                init_fn(m.weight, *args, **kwargs)
                srg += ', weight: '+init_fn.__name__

            if (key in b_kw or 'Default' in b_kw) and hasattr(m, 'bias') and m.bias is not None:
                init_fn, args, kwargs = b_kw[key] if key in b_kw else b_kw['Default']
                init_fn(m.bias, *args, **kwargs)
                srg += ', bias: '+init_fn.__name__
            
            if pr: print(srg)




