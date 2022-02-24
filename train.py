from gnn4ifa.data import IfaDataset

dataset = IfaDataset(root='/Users/andrea.agiollo/Documents/PhD/Projects/GNN-x-IFA/ifa_data_tg',
                     download_folder='/Users/andrea.agiollo/Documents/PhD/Projects/GNN-x-IFA/ifa_data')
print('dataset.raw_dir: {}'.format(dataset.raw_dir))
print('dataset.processed_dir: {}'.format(dataset.processed_dir))