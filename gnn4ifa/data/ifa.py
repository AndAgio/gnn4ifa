import os
import random
import shutil
import zipfile
import time
import glob
import torch
from torch_geometric.data import InMemoryDataset, download_url
from torch_geometric.loader import DataLoader
# Import modules
from gnn4ifa.utils import download_file_from_google_drive
from .extractor import Extractor


class IfaDataset(InMemoryDataset):
    def __init__(self,
                 root='ifa_data_tg',
                 download_folder='ifa_data',
                 transform=None,
                 pre_transform=None,
                 scenario='existing',
                 topology='small',
                 train_sim_ids=[1, 2, 3],
                 val_sim_ids=[4],
                 test_sim_ids=[5],
                 simulation_time=300,
                 time_att_start=50,
                 split='train'):
        self.download_folder = download_folder
        assert scenario in ['existing', 'non_existing', 'normal']
        self.scenario = scenario
        assert topology in ['small', 'dfn']
        self.topology = topology
        for train_sim_id in train_sim_ids:
            assert 1 <= train_sim_id <= 5
        for val_sim_id in val_sim_ids:
            assert 1 <= val_sim_id <= 5
        for test_sim_id in test_sim_ids:
            assert 1 <= test_sim_id <= 5
        assert set(train_sim_ids + val_sim_ids + test_sim_ids) == {1, 2, 3, 4, 5}
        self.train_sim_ids = train_sim_ids
        self.val_sim_ids = val_sim_ids
        self.test_sim_ids = test_sim_ids
        self.simulation_time = simulation_time
        self.time_att_start = time_att_start
        self.root = root
        print('self.root: {}'.format(self.root))
        super(IfaDataset, self).__init__(self.root, transform, pre_transform)
        if split == 'train':
            path = self.processed_paths[0]
        elif split == 'val':
            path = self.processed_paths[1]
        elif split == 'test':
            path = self.processed_paths[2]
        else:
            raise ValueError(f"Split '{split}' found, but expected either "
                             f"'train', 'val', or 'test'")
        self.data, self.slices = torch.load(path)

    @property
    def download_dir(self) -> str:
        return os.path.join(self.download_folder,
                            'IFA_4_{}'.format(self.scenario) if self.scenario != 'normal' else self.scenario,
                            '{}_topology'.format(self.topology) if self.topology != 'dfn' else '{}_topology'.format(
                                self.topology.upper()))

    @property
    def raw_dir(self) -> str:
        return os.path.join(self.root, self.scenario, self.topology, 'raw')

    @property
    def processed_dir(self) -> str:
        return os.path.join(self.root, self.scenario, self.topology, 'processed')

    @property
    def download_file_names(self):
        if self.scenario == 'existing' and self.topology == 'dfn':
            frequencies = ['4x', '8x', '16x', '32x']
        elif self.scenario == 'existing' and self.topology == 'small':
            frequencies = ['4x', '8x', '16x', '32x', '64x']
        elif self.scenario == 'non_existing' and self.topology == 'dfn':
            raise FileNotFoundError('Scenario {} and topology {} are incompatible at the moment'.format(self.scenario,
                                                                                                        self.topology))
        elif self.scenario == 'non_existing' and self.topology == 'small':
            frequencies = ['16x', '32x', '64x', '128x', '256x']
        elif self.scenario == 'normal':
            frequencies = None
        else:
            raise ValueError('Something wrong with scenario {} and topology {}'.format(self.scenario,
                                                                                       self.topology))
        # Define files that should be available as raw in the dataset
        names = ['drop-trace', 'pit-size', 'rate-trace']
        if frequencies:
            file_names = ['{}/{}-{}.txt'.format(freq, name, index) for freq in frequencies for name in names for index
                          in range(1, 6)]
            print('file_names are: {}'.format(file_names))
        else:
            file_names = ['{}-{}.txt'.format(name, index) for name in names for index in range(1, 6)]
            print('file_names are: {}'.format(file_names))
        return file_names

    @property
    def raw_file_names(self):
        return ['train_{}_data.pt'.format(self.train_sim_ids),
                'val_{}_data.pt'.format(self.val_sim_ids),
                'test_{}_data.pt'.format(self.test_sim_ids)]

    @property
    def processed_file_names(self):
        return ['train_{}_data.pt'.format(self.train_sim_ids),
                'val_{}_data.pt'.format(self.val_sim_ids),
                'test_{}_data.pt'.format(self.test_sim_ids)]

    def download(self, force=False):
        # Download dataset only if the download folder is not found
        if not os.path.exists(self.download_dir) or force:
            raise NotImplementedError('Not yet moved dataset to shared drive folder')
            # Download tfrecord file if not found...
            print('Downloading dataset file, this will take a while...')
            radar_online = '1uJ9HTlduxTfSnz91-n8_8-fleUgkUPWB'
            tmp_download_folder = os.path.join(os.getcwd(), 'dwn')
            if not os.path.exists(tmp_download_folder):
                os.makedirs(tmp_download_folder)
            download_file_from_google_drive(radar_online, os.path.join(tmp_download_folder, 'RADAR.zip'))
            # Extract zip files from downloaded dataset
            zf = zipfile.ZipFile(os.path.join(tmp_download_folder, 'RADAR.zip'), 'r')
            print('Unzipping dataset...')
            zf.extractall(tmp_download_folder)
            # Make order into the project folder moving extracted dataset into home and removing temporary download folder
            print('Moving dataset to clean repo...')
            shutil.move(os.path.join(tmp_download_folder, 'RADAR'), os.path.join(os.getcwd(), self.download_folder))
            shutil.rmtree(tmp_download_folder)
            # Run _preprocess to activate the extractor which converts the dataset files into tg_graphs
            self.convert_dataset_to_tg_graphs()

    def convert_dataset_to_tg_graphs(self):
        # Import stored dictionary of data
        file_names = glob.glob(os.path.join(self.download_dir, '*', '*.txt'))
        Extractor(data_dir=self.download_folder,
                  scenario=self.scenario,
                  topology=self.topology,
                  train_sim_ids=self.train_sim_ids,
                  val_sim_ids=self.val_sim_ids,
                  test_sim_ids=self.test_sim_ids,
                  simulation_time=self.simulation_time,
                  time_att_start=self.time_att_start).run(downloaded_data_file=file_names,
                                                          raw_dir=self.raw_dir,
                                                          raw_file_names=self.raw_file_names)

    def process(self):
        # Check if it is possible to load the tg raw file
        try:
            data_list = torch.load(os.path.join(self.raw_dir, self.raw_file_names[0]))
        except FileNotFoundError:
            # Check if the dataset is already downloaded or not
            # Gather real names of files
            existing_file_names = glob.glob(os.path.join(self.download_dir, '*', '*.txt'))
            # If the two sets don't match it means that the dataset was not downloaded yet
            required_file_names = [os.path.join(self.download_dir, name) for name in self.download_file_names]
            print('required_file_names: {}'.format(required_file_names))
            print('existing_file_names: {}'.format(existing_file_names))
            print('self.download_dir: {}'.format(self.download_dir))
            if set(required_file_names) != set(existing_file_names):
                print('Didn\'t find the dataset. Downloading it...')
                self.download()
                print('Running the extractor...')
                self.convert_dataset_to_tg_graphs()
            else:
                print('Tg raw data not found, but dataset was found. Running the extractor...')
                self.convert_dataset_to_tg_graphs()
        # Iterate over the splits, load raw data, filter and transform them
        for index in range(len(self.raw_file_names)):
            # Load the raw tg_data
            data_list = torch.load(os.path.join(self.raw_dir, self.raw_file_names[index]))
            # Apply pre_filter and pre_transform if necessary
            if self.pre_filter is not None:
                data_list = [data for data in data_list if self.pre_filter(data)]
            if self.pre_transform is not None:
                data_list = [self.pre_transform(data) for data in data_list]
            # Store
            self.store_processed_data(data_list, self.processed_paths[index])

    def store_processed_data(self, data_list, name):
        data, slices = self.collate(data_list)
        torch.save((data, slices), name)

    def get_scenario_data(self, scenario='Legitimate'):
        # Return only graphs belonging to a specific scenario
        scenario_label = get_scenario_labels_dict()[scenario]
        data = [data for data in self if data['scenario'] == scenario_label]
        print('Number of filtered graphs over {} scenario: {}'.format(scenario, len(data)))
        return data

    def get_all_attack_data(self, attack='Blackhole'):
        # Return all graphs where a specific attack is active
        # Filter graphs by scenario and get only those graphs that have attack_is_on==True
        whole_scenario_data = self.get_scenario_data(scenario=attack)
        data = [data for data in whole_scenario_data if data['attack_is_on'] == 1]
        print('Number of filtered graphs over active {} attack: {}'.format(attack, len(data)))
        # print('Data example: {} -> scenario: {} -> attack_is_on: {}'.format(data[0], data[0].scenario, data[0].attack_is_on))
        # print('Data example: {} -> scenario: {} -> attack_is_on: {}'.format(data[len(data)-1], data[len(data)-1].scenario, data[len(data)-1].attack_is_on))
        # print('Data example: {} -> scenario: {} -> attack_is_on: {}'.format(data[int(len(data)/2)], data[int(len(data)/2)].scenario, data[int(len(data)/2)].attack_is_on))
        return data

    def get_all_legitimate_data(self):
        # Return all graphs where no attack is active
        # Gather all graphs over all scenarios and get only those graphs that have attack_is_on==False
        data = [data for data in self]
        print('Number of non-filtered graphs: {}'.format(len(data)))
        data = [data for data in self if data['attack_is_on'] == 0]
        print('Number of filtered graphs: {}'.format(len(data)))
        # print('Data example: {} -> scenario: {} -> attack_is_on: {}'.format(data[0], data[0].scenario, data[0].attack_is_on))
        # print('Data example: {} -> scenario: {} -> attack_is_on: {}'.format(data[len(data)-1], data[len(data)-1].scenario, data[len(data)-1].attack_is_on))
        # print('Data example: {} -> scenario: {} -> attack_is_on: {}'.format(data[int(len(data)/2)], data[int(len(data)/2)].scenario, data[int(len(data)/2)].attack_is_on))
        return data
