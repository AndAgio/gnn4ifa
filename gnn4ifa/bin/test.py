import os

import numpy as np
import torch
import torch_geometric as tg
# Import my modules
from gnn4ifa.data import IfaDataset
from gnn4ifa.transforms import NormalizeFeatures, RandomNodeMasking
from gnn4ifa.models import Classifier, AutoEncoder
from gnn4ifa.metrics import Accuracy, F1Score, MSE, MAE
from gnn4ifa.utils import get_data_loader


class Tester():
    def __init__(self,
                 dataset_folder='ifa_data_tg',
                 download_dataset_folder='ifa_data',
                 train_scenario='existing',
                 train_topology='small',
                 test_scenario='existing',
                 test_topology='small',
                 frequencies=None,
                 train_sim_ids=[1, 2, 3],
                 val_sim_ids=[4],
                 test_sim_ids=[5],
                 simulation_time=300,
                 time_att_start=50,
                 chosen_model='class_gcn_2x100_mean',
                 masking=False,
                 percentile=0.99,
                 out_path='outputs'):
        # Dataset related variables
        self.dataset_folder = dataset_folder
        self.download_dataset_folder = download_dataset_folder
        self.train_scenario = train_scenario
        self.train_topology = train_topology
        self.test_scenario = test_scenario
        self.test_topology = test_topology
        self.frequencies = frequencies
        self.train_sim_ids = train_sim_ids
        self.val_sim_ids = val_sim_ids
        self.test_sim_ids = test_sim_ids
        self.simulation_time = simulation_time
        self.time_att_start = time_att_start
        # Model related variables
        if chosen_model.split('_')[0] == 'class':
            self.mode = 'class'
            # Metrics related features
            self.chosen_metrics = ['acc', 'f1']
            self.metric_to_check = 'acc'
            self.lr_metric_to_check = 'acc'
        elif chosen_model.split('_')[0] == 'anomaly':
            self.mode = 'anomaly'
            # Metrics related features
            self.chosen_metrics = ['mse', 'mae']
            self.metric_to_check = 'mse'
            self.lr_metric_to_check = 'mse'
            self.masking = masking
            self.percentile = percentile
        else:
            raise ValueError('Model should indicate training mode between classification and anomaly!')
        self.chosen_model = chosen_model
        # Training related variables
        self.out_path = os.path.join(os.getcwd(), out_path)
        self.trained_models_folder = os.path.join(self.out_path, 'trained_models')
        # Check if GPU is available
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # Get dataset and setup the trainer
        self.get_dataset()
        self.get_model()

    def get_dataset(self):
        # Get dataset
        if self.mode == 'class':
            transform = tg.transforms.Compose([NormalizeFeatures(attrs=['x'])])
        elif self.mode == 'anomaly':
            if self.masking:
                transform = tg.transforms.Compose([NormalizeFeatures(attrs=['x']),
                                                   RandomNodeMasking(sampling_strategy='local',
                                                                     sampling_probability=0.3)
                                                   ])
            else:
                transform = tg.transforms.Compose([NormalizeFeatures(attrs=['x'])])
        else:
            raise ValueError('Something wrong with selected mode between classification and anomaly!')
        # Training
        self.train_dataset = IfaDataset(root=self.dataset_folder,
                                        download_folder=self.download_dataset_folder,
                                        transform=transform,
                                        scenario=self.train_scenario if self.mode == 'class' else 'normal',
                                        topology=self.train_topology,
                                        train_sim_ids=self.train_sim_ids,
                                        val_sim_ids=self.val_sim_ids,
                                        test_sim_ids=self.test_sim_ids,
                                        simulation_time=self.simulation_time,
                                        time_att_start=self.time_att_start,
                                        split='train')
        print('Number of training examples: {}'.format(len(self.train_dataset)))
        # Test
        self.test_dataset = IfaDataset(root=self.dataset_folder,
                                       download_folder=self.download_dataset_folder,
                                       transform=transform,
                                       scenario=self.test_scenario,
                                       topology=self.test_topology,
                                       train_sim_ids=self.train_sim_ids,
                                       val_sim_ids=self.val_sim_ids,
                                       test_sim_ids=self.test_sim_ids,
                                       simulation_time=self.simulation_time,
                                       time_att_start=self.time_att_start,
                                       split='test')
        print('Number of test examples: {}'.format(len(self.test_dataset)))
        self.test_loader = get_data_loader(
            self.test_dataset if self.mode == 'class' else self.test_dataset.get_all_data(frequencies=self.frequencies),
            batch_size=1,
            shuffle=False)
        # Get the number of node features
        self.num_node_features = self.train_dataset.num_features
        print('self.num_node_features:', self.num_node_features)

    def get_model(self):
        # Get the model depending on the string passed by user
        self.load_best_model()
        # Move model to GPU or CPU
        self.model = self.model.to(self.device)
        # Setup metrics depending on the choice
        self.metrics = {'acc': Accuracy(),
                        'f1': F1Score()}

    def load_best_model(self):
        if self.mode == 'class':
            model_name = '{} sce_{} topo_{} best.pt'.format(self.chosen_model,
                                                            self.train_scenario,
                                                            self.train_topology)
        elif self.mode == 'anomaly':
            model_name = '{} mask_{} sce_{} topo_{} best.pt'.format(self.chosen_model,
                                                                    self.masking,
                                                                    self.train_scenario,
                                                                    self.train_topology)
        else:
            raise ValueError('Something went wrong with mode selection')
        model_path = os.path.join(self.trained_models_folder, model_name)
        self.model = torch.load(model_path)
        # Move model to GPU or CPU
        self.model = self.model.to(self.device)

    def run(self):
        print('Start testing...')
        self.test()

    @torch.no_grad()
    def test(self):
        # Set the valuer to be non trainable
        self.model.eval()
        if self.mode == 'class':
            self._test_class()
        elif self.mode == 'anomaly':
            self._test_anomaly()
        else:
            raise ValueError('Something wrong with mode selection')

    @torch.no_grad()
    def _test_class(self):
        all_preds = None
        all_labels = None
        for batch_index, data in enumerate(self.test_loader):
            batch_preds, batch_labels = self.test_step(data)
            # Append batch predictions and labels to the list containing every prediction and every label
            if batch_index == 0:
                all_preds = batch_preds
                all_labels = batch_labels
            else:
                all_preds = torch.cat((all_preds, batch_preds), dim=0)
                all_labels = torch.cat((all_labels, batch_labels), dim=0)
            # Compute metrics over predictions
            scores = {}
            for metric_name, metric_object in self.metrics.items():
                scores[metric_name] = metric_object.compute(y_pred=all_preds,
                                                            y_true=all_labels)
            self.print_test_message(index_batch=batch_index, metrics=scores)
        print()

    @torch.no_grad()
    def _test_anomaly(self):
        threshold_metric = 'mse'
        # Get threshold from training
        threshold = self.get_threshold(metric=threshold_metric)
        all_preds = None
        all_labels = None
        for batch_index, data in enumerate(self.test_loader):
            batch_preds, batch_labels = self.test_step(data, threshold, threshold_metric=threshold_metric)
            # Append batch predictions and labels to the list containing every prediction and every label
            if batch_index == 0:
                all_preds = batch_preds
                all_labels = batch_labels
            else:
                all_preds = torch.cat((all_preds, batch_preds), dim=0)
                all_labels = torch.cat((all_labels, batch_labels), dim=0)
            # Compute metrics over predictions
            scores = {}
            for metric_name, metric_object in self.metrics.items():
                scores[metric_name] = metric_object.compute(y_pred=all_preds,
                                                            y_true=all_labels)
            self.print_test_message(index_batch=batch_index, metrics=scores)
        print()

    @torch.no_grad()
    def get_threshold(self, metric='mae'):
        print('Computing {}s over legitimate training samples'.format(metric.upper()))
        values = []
        loader = get_data_loader(self.train_dataset.get_all_legitimate_data(),
                                 batch_size=1,
                                 shuffle=True)
        for batch_index, data in enumerate(loader):
            data = data.to(self.device)
            # Pass graphs through model
            y_pred = self.model(data)

            if not self.masking:
                # Get labels from data
                y_true = data.x.float()
            else:
                # Get masks for masked nodes and edges to be used to optimize model
                nodes_mask = data.masked_nodes_indices
                # Get labels for node and edge predictions
                y_true = data.original_x[nodes_mask].float()
                # Mask the predictions over the masked nodes only
                y_pred = y_pred[nodes_mask]

            if metric == 'mae':
                values_samples = MAE.compute(y_pred=y_pred, y_true=y_true).item()
            elif metric == 'mse':
                values_samples = MSE.compute(y_pred=y_pred, y_true=y_true).item()
            else:
                raise ValueError('Metric {} is not available for computing threshold')
            # print('values_samples: {}'.format(values_samples))
            values.append(values_samples)
        # Get threshold value depending on the percentile given
        sorted_values = np.sort(values)
        threshold_index = int(len(sorted_values) * self.percentile)
        threshold = sorted_values[threshold_index]
        print('sorted_values: {}'.format(sorted_values))
        print('{} threshold obtained: {}'.format(metric.upper(), threshold))
        return threshold

    @torch.no_grad()
    def test_step(self, data, threshold=None, threshold_metric='mae'):
        data = data.to(self.device)
        if self.mode == 'anomaly':
            # Pass graphs through model
            preds = self.model(data)

            if not self.masking:
                # Get labels from data
                y_true = data.x.float()
            else:
                # Get masks for masked nodes and edges to be used to optimize model
                nodes_mask = data.masked_nodes_indices
                # Get labels for node and edge predictions
                y_true = data.original_x[nodes_mask].float()
                # Mask the predictions over the masked nodes only
                preds = preds[nodes_mask]

            if threshold_metric == 'mae':
                preds_mses = MAE.compute(y_pred=preds, y_true=y_true)
            elif threshold_metric == 'mse':
                preds_mses = MSE.compute(y_pred=preds, y_true=y_true)
            else:
                raise ValueError('Metric {} is not available for computing threshold')
            y_pred = torch.as_tensor(np.array([preds_mses > threshold]).astype('int'))
            # print('preds: {} -> original_x: {}'.format(preds,
            #                                            y_true))
            print('preds_mses: {:.5f} -> label: {} -> '
                  'prediction: {} -> freq: {}'.format(preds_mses,
                                                      data.attack_is_on.item(),
                                                      y_pred.item(),
                                                      data.frequency.item()))
        elif self.mode == 'class':
            y_pred = self.model(data)
        # Get labels from data
        y_true = data.attack_is_on
        # Return predictions and labels
        return y_pred, y_true

    def print_test_message(self, index_batch, metrics):
        message = '| '
        bar_length = 10
        total_batches = len(self.test_loader)
        progress = float(index_batch) / float(total_batches)
        if progress >= 1.:
            progress = 1
        block = int(round(bar_length * progress))
        message += '[{}] | TEST: '.format('=' * block + ' ' * (bar_length - block))
        if metrics is not None:
            metrics_message = ''
            for metric_name, metric_value in metrics.items():
                metrics_message += '{}={:.5f} '.format(metric_name,
                                                       metric_value)
            message += metrics_message
        message += '|'
        print(message, end='\r')
