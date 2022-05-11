from torch_geometric.loader import DataLoader


def get_scenario_labels_dict():
    labels_dict = {'normal': 0,
                   'existing': 1,
                   'non_existing': 2}
    return labels_dict


def get_labels_scenario_dict():
    labels_dict = {0: 'normal',
                   1: 'existing',
                   2: 'non_existing'}
    return labels_dict


def get_attacker_type_labels_dict():
    labels_dict = {'none': -1,
                   'fixed': 0,
                   'variable': 1}
    return labels_dict


def get_labels_attacker_type_dict():
    labels_dict = {-1: 'none',
                   0: 'fixed',
                   1: 'variable'}
    return labels_dict


def get_data_loader(dataset, batch_size=32, shuffle=False):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return loader
