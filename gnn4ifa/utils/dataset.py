from torch_geometric.loader import DataLoader


def get_scenario_labels_dict():
    labels_dict = {'normal': 0,
                   'existing': 1,
                   'non_existing': 2}
    return labels_dict


def get_data_loader(dataset, batch_size=32, shuffle=False):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return loader