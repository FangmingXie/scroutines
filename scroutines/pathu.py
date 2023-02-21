import os

datasets = {
    'tag': 'path',
}

def get_path(key, datasets=datasets, check=False):
    """
    Retrieve path from a key; or the key itself 
    """
    if key in datasets.keys():
        path = datasets[key]
    else:
        path = key
    
    if check:
        assert os.path.isfile(path)
    return path