import numpy as np

def sample_config(ranges, N=1, keys=None):
    r'''Given a dictionary of ranges, sample N random permutations'''
    if keys is None:
        keys = ranges.keys()
    result = dict()
    for key in keys:
        subelement = dict()
        for subkey, subvalues in ranges[key].items():
            subvalues = np.array(subvalues)
            if subvalues.ndim == 1:
                subelement[subkey] = np.random.choice(subvalues, size=N, replace=True)
            else:
                indices = np.arange(len(subvalues))
                indices = np.random.choice(indices, size=N, replace=True)
                subelement[subkey] = subvalues[indices]
        result[key] = subelement
    return result

def unroll_samples(samples, merge_x_y=True, defaults=None):
    r'''Given a dictionary of samples, unroll them into a list of dictionaries
    
    The format of the dictionary is assumed to be 'key': {'subkey': [value, value, ...]}
    The converted format is a list of dictionaries, where each dictionary has the same keys, but the values are single values instead of lists
    [{'key': {'subkey': value}}, {'key': {'subkey': value}}, ...]
    '''
    N = len(samples['eyes']['shape'])
    defaults = defaults or dict()
    result = [dict() for _ in range(N)]
    for key, value in samples.items():
        value_defaults = defaults.get(key, dict())
        for subkey, default_value in value_defaults.items():
            for idx in range(N):
                result[idx][key] = result[idx].get(key, dict())
                result[idx][key][subkey] = default_value
        for subkey, subvalues in value.items():
            for idx in range(N):
                result[idx][key] = result[idx].get(key, dict())
                result[idx][key][subkey] = subvalues[idx]
    if merge_x_y:
        for element in result:
            for key, value in element.items():
                seen = set()
                for subkey in value.keys():
                    if subkey in seen:
                        continue
                    if subkey.endswith('_x'):
                        newkey = subkey[:-2]
                        ykey = newkey + '_y'
                        if ykey in value.keys():
                            seen.add(subkey)
                for xkey in seen:
                    newkey = xkey[:-2]
                    ykey = newkey + '_y'
                    value[newkey] = (value[xkey], value[ykey])
                    del value[xkey]
                    del value[ykey]
    return result


def _create_min_max_step(d: dict, to_numpy: bool = False):
    seen = dict()
    for key, value in d.items():
        if isinstance(value, dict):
            if 'start' in value and 'stop' in value:
                if 'num' in value:
                    value = np.linspace(value['start'], value['stop'], value['num'])
                else:
                    value = np.linspace(value['start'], value['stop'], 10)
            elif 'min' in value and 'max' in value:
                if 'step' in value:
                    value = np.arange(value['min'], value['max'], value['step'])
                else:
                    value = np.arange(value['min'], value['max'])
            else:
                value = _create_min_max_step(value, to_numpy=to_numpy)
        elif isinstance(value, (list, tuple)):
            value = np.array(value)
        seen[key] = value
    # for key, value in seen.items():
        # d[key] = value
    return seen


def read_ranges(filename):
    r'''Reads the ranges from a YAML file, and creates a dictionary of ranges or possible values.
    
    Key rules:
        1. If the value is a dict, and has only 'start', 'stop', and optionally 'num' ==> Use linspace(start, stop, num)
        2. If the value is a dict, and has only 'min', 'max', and optionally 'step' ==> Use arange(min, max, step)
    '''
    import yaml
    with open(filename, 'r') as f:
        ranges = yaml.safe_load(f)
    ranges = _create_min_max_step(ranges, to_numpy=True)
    return ranges
