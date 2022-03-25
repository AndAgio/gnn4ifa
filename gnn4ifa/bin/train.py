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


class Trainer():
    def __init__(self,
                 dataset_folder='ifa_data_tg',
                 download_dataset_folder='ifa_data',
                 train_scenario='existing',
                 train_topology='small',
                 frequencies=None,
                 train_sim_ids=[1, 2, 3],
                 val_sim_ids=[4],
                 test_sim_ids=[5],
                 simulation_time=300,
                 time_att_start=50,
                 differential=False,
                 chosen_model='class_gcn_2x100_mean',
                 masking=False,
                 percentile=0.99,
                 optimizer='sgd',
                 momentum=0.9,
                 weight_decay=5e-4,
                 batch_size=32,
                 epochs=100,
                 lr=0.01,
                 out_path='outputs'):
        # Dataset related variables
        self.dataset_folder = dataset_folder
        self.download_dataset_folder = download_dataset_folder
        self.train_scenario = train_scenario
        self.train_topology = train_topology
        self.frequencies = frequencies
        self.train_sim_ids = train_sim_ids
        self.val_sim_ids = val_sim_ids
        self.test_sim_ids = test_sim_ids
        self.simulation_time = simulation_time
        self.time_att_start = time_att_start
        if differential and chosen_model.split('_')[0] == 'class':
            raise ValueError('Differential is not available for classification model')
        self.differential = differential
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
        self.chosen_optimizer = optimizer
        self.batch_size = batch_size
        self.learning_rate = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.out_path = os.path.join(os.getcwd(), out_path)
        self.trained_models_folder = os.path.join(self.out_path, 'trained_models')
        # Check if GPU is available
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # Get dataset and setup the trainer
        self.get_dataset()
        self.setup()

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
                                        differential=self.differential,
                                        split='train')
        print('self.train_dataset[0]: {}'.format(self.train_dataset[0]))
        print('Number of training examples: {}'.format(len(self.train_dataset)))
        self.train_loader = get_data_loader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True)
        # Validation
        self.val_dataset = IfaDataset(root=self.dataset_folder,
                                      download_folder=self.download_dataset_folder,
                                      transform=transform,
                                      scenario=self.train_scenario if self.mode == 'class' else 'normal',
                                      topology=self.train_topology,
                                      train_sim_ids=self.train_sim_ids,
                                      val_sim_ids=self.val_sim_ids,
                                      test_sim_ids=self.test_sim_ids,
                                      simulation_time=self.simulation_time,
                                      time_att_start=self.time_att_start,
                                      differential=self.differential,
                                      split='val')
        print('Number of validation examples: {}'.format(len(self.val_dataset)))
        self.val_loader = get_data_loader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=True)
        # Test
        self.test_dataset = IfaDataset(root=self.dataset_folder,
                                       download_folder=self.download_dataset_folder,
                                       transform=transform,
                                       scenario=self.train_scenario,
                                       topology=self.train_topology,
                                       train_sim_ids=self.train_sim_ids,
                                       val_sim_ids=self.val_sim_ids,
                                       test_sim_ids=self.test_sim_ids if self.mode == 'class' else [1, 2, 3, 4, 5],
                                       simulation_time=self.simulation_time,
                                       time_att_start=self.time_att_start,
                                       differential=self.differential,
                                       split='test')
        print('Number of test examples: {}'.format(len(self.test_dataset)))
        if self.mode == 'class':
            self.test_loader = get_data_loader(self.test_dataset,
                                               batch_size=1,
                                               shuffle=False)
        elif self.differential:
            self.test_loader = self.test_dataset.get_data_dict(frequencies=self.frequencies)
        elif not self.differential:
            self.test_loader = get_data_loader(self.test_dataset.get_all_data(frequencies=self.frequencies),
                                               batch_size=1,
                                               shuffle=False)
        else:
            raise ValueError('Something wrong with test set loading!')
        # Get the number of node features
        self.num_node_features = self.train_dataset.num_features
        print('self.num_node_features:', self.num_node_features)

    def setup(self):
        # Get the model depending on the string passed by user
        if self.mode == 'class':
            self.model = Classifier(input_node_dim=self.num_node_features,
                                    conv_type=self.chosen_model.split('_')[1],
                                    hidden_dim=int(self.chosen_model.split('_')[2].split('x')[-1]),
                                    n_layers=int(self.chosen_model.split('_')[2].split('x')[0]),
                                    pooling_type=self.chosen_model.split('_')[3],
                                    n_classes=2)
            # Define criterion for loss
            self.criterion = torch.nn.CrossEntropyLoss()
            # Setup metrics depending on the choice
            self.metrics = {}
            for metric in self.chosen_metrics:
                if metric == 'acc':
                    self.metrics[metric] = Accuracy()
                elif metric == 'f1':
                    self.metrics[metric] = F1Score()
                else:
                    raise ValueError('The metric {} is not available in classification mode!'.format(metric))
        elif self.mode == 'anomaly':
            self.model = AutoEncoder(input_node_dim=self.num_node_features,
                                     conv_type=self.chosen_model.split('_')[1],
                                     hidden_dim=int(self.chosen_model.split('_')[2].split('x')[-1]),
                                     n_encoding_layers=int(self.chosen_model.split('_')[2].split('x')[0]),
                                     n_decoding_layers=int(self.chosen_model.split('_')[2].split('x')[1]))
            # Define criterion for loss
            self.criterion = torch.nn.MSELoss()
            # Setup metrics depending on the choice
            self.metrics = {}
            for metric in self.chosen_metrics:
                if metric == 'mse':
                    self.metrics[metric] = MSE()
                elif metric == 'mae':
                    self.metrics[metric] = MAE()
                else:
                    raise ValueError('The metric {} is not available in our implementation yet!'.format(metric))
        else:
            raise ValueError('Something wrong with selected mode between classification and anomaly!')
        # Move model to GPU or CPU
        self.model = self.model.to(self.device)
        # Get the optimizer depending on the selected one
        if self.chosen_optimizer == 'sgd':
            self.optimizer = torch.optim.SGD(self.model.parameters(),
                                             lr=self.learning_rate,
                                             momentum=self.momentum,
                                             weight_decay=self.weight_decay)
        elif self.chosen_optimizer == 'adam':
            self.optimizer = torch.optim.Adam(self.model.parameters(),
                                              lr=self.learning_rate,
                                              weight_decay=self.weight_decay)
        else:
            raise ValueError('The optimizer you selected ({}) is not available!'.format(self.chosen_optimizer))
        # Setup learning rate scheduler
        self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer,
                                                                       mode='min',
                                                                       factor=0.8,
                                                                       patience=10,
                                                                       min_lr=0.00001)

    def save_best_model(self):
        # Check if directory for trained models exists, if not make it
        if not os.path.exists(self.trained_models_folder):
            os.makedirs(self.trained_models_folder)
        if self.mode == 'class':
            model_name = '{} sce_{} topo_{} diff_{} best.pt'.format(self.chosen_model,
                                                                    self.train_scenario,
                                                                    self.train_topology,
                                                                    self.differential)
        elif self.mode == 'anomaly':
            model_name = '{} mask_{} sce_{} topo_{} diff_{} best.pt'.format(self.chosen_model,
                                                                            self.masking,
                                                                            self.train_scenario,
                                                                            self.train_topology,
                                                                            self.differential)
        else:
            raise ValueError('Something went wrong with mode selection')
        model_path = os.path.join(self.trained_models_folder, model_name)
        torch.save(self.model.cpu(), model_path)

    def load_best_model(self):
        if self.mode == 'class':
            model_name = '{} sce_{} topo_{} diff_{} best.pt'.format(self.chosen_model,
                                                                    self.train_scenario,
                                                                    self.train_topology,
                                                                    self.differential)
        elif self.mode == 'anomaly':
            model_name = '{} mask_{} sce_{} topo_{} diff_{} best.pt'.format(self.chosen_model,
                                                                            self.masking,
                                                                            self.train_scenario,
                                                                            self.train_topology,
                                                                            self.differential)
        else:
            raise ValueError('Something went wrong with mode selection')
        model_path = os.path.join(self.trained_models_folder, model_name)
        self.model = torch.load(model_path)
        # Move model to GPU or CPU
        self.model = self.model.to(self.device)

    def run(self, print_examples=False):
        print('Start training...')
        # Define best metric to store best model
        best_met = 0.0
        # Iterate over the number of epochs defined in the init
        for epoch in range(self.epochs):
            # Train
            train_loss, train_metrics = self.train_epoch(epoch)
            # Validate
            val_loss, val_metrics = self.val_epoch(epoch, train_loss, train_metrics)
            print()
            # Save best model if metric improves
            if val_metrics[self.metric_to_check] > best_met:
                best_met = val_metrics[self.metric_to_check]
                self.save_best_model()
            # Update learning rate depending on the scheduler
            if self.lr_metric_to_check == 'loss':
                self.lr_scheduler.step(train_loss)
            else:
                self.lr_scheduler.step(val_metrics[self.lr_metric_to_check])
        print('Finished Training. Testing...')
        self.load_best_model()
        self.test()

    def train_epoch(self, epoch):
        # Set the valuer to be trainable
        self.model.train()
        avg_loss, avg_metrics = self._train_epoch(epoch)
        return avg_loss, avg_metrics

    def _train_epoch(self, epoch):
        running_loss = 0.0
        running_scores = {met_name: 0.0 for met_name in self.metrics.keys()}
        for batch_index, data in enumerate(self.train_loader):
            batch_loss, batch_scores = self.train_step(data)
            running_loss += batch_loss
            for metric_name, metric_value in batch_scores.items():
                running_scores[metric_name] += metric_value
            avg_loss = running_loss / (batch_index + 1)
            avg_metrics = {met_name: met_value / (batch_index + 1) for met_name, met_value in running_scores.items()}
            self.print_message(epoch,
                               index_train_batch=batch_index,
                               train_loss=avg_loss,
                               train_mets=avg_metrics,
                               index_val_batch=None,
                               val_loss=None,
                               val_mets=None)
        return avg_loss, avg_metrics

    def train_step(self, data):
        data = data.to(self.device)
        # print('data: {}'.format(data))
        # print('data.x: {}'.format(data.x))
        # print('data.attack_is_on: {}'.format(data.attack_is_on))
        # zero the parameter gradients
        self.optimizer.zero_grad()
        # Pass graphs through model
        y_pred = self.model(data)
        # Get labels from data
        if self.mode == 'class':
            y_true = data.attack_is_on.long()
        elif self.mode == 'anomaly':

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

        else:
            raise ValueError('Something wrong with selected mode between classification and anomaly!')
        # print('data: {} -> data.x: {} -> pred: {} -> y_true: {}'.format(data, data.x, y_pred, y_true))
        # Compute reconstruction loss
        loss = self.criterion(target=y_true,
                              input=y_pred)
        if torch.isnan(loss).any():
            raise ValueError('Something very wrong! Loss turned NaN!')
        # Compute gradient
        loss.backward()
        # Backpropragate
        self.optimizer.step()
        # Compute metrics over predictions
        scores = {}
        for metric_name, metric_object in self.metrics.items():
            scores[metric_name] = metric_object.compute(y_pred=y_pred,
                                                        y_true=y_true)
        # Return loss and metrics
        return loss.item(), scores

    @torch.no_grad()
    def val_epoch(self, epoch, train_loss, train_mets):
        # Set the valuer to be non trainable
        self.model.eval()
        avg_loss, avg_metrics = self._val_epoch(epoch, train_loss, train_mets)
        return avg_loss, avg_metrics

    @torch.no_grad()
    def _val_epoch(self, epoch, train_loss, train_mets):
        running_loss = 0.0
        running_scores = {met_name: 0.0 for met_name in self.metrics.keys()}
        for batch_index, data in enumerate(self.val_loader):
            batch_loss, batch_scores = self.val_step(data)
            running_loss += batch_loss
            for metric_name, metric_value in batch_scores.items():
                running_scores[metric_name] += metric_value
            avg_loss = running_loss / (batch_index + 1)
            avg_metrics = {met_name: met_value / (batch_index + 1) for met_name, met_value in running_scores.items()}
            self.print_message(epoch,
                               index_train_batch=len(self.train_loader),
                               train_loss=train_loss,
                               train_mets=train_mets,
                               index_val_batch=batch_index,
                               val_loss=avg_loss,
                               val_mets=avg_metrics)
        return avg_loss, avg_metrics

    @torch.no_grad()
    def val_step(self, data):
        data = data.to(self.device)
        # zero the parameter gradients
        self.optimizer.zero_grad()
        # Pass graphs through model
        y_pred = self.model(data)
        # Get labels from data
        if self.mode == 'class':
            y_true = data.attack_is_on.long()
        elif self.mode == 'anomaly':

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

        else:
            raise ValueError('Something wrong with selected mode between classification and anomaly!')
        # Compute reconstruction loss
        loss = self.criterion(target=y_true,
                              input=y_pred)
        # Compute metrics over predictions
        scores = {}
        for metric_name, metric_object in self.metrics.items():
            scores[metric_name] = metric_object.compute(y_pred=y_pred,
                                                        y_true=y_true)
        # Return loss and metrics
        return loss.item(), scores

    @torch.no_grad()
    def test(self):
        # Set the valuer to be non trainable
        self.model.eval()
        if self.mode == 'class':
            self._test_class()
        elif self.mode == 'anomaly' and self.differential:
            self._test_anomaly_differential()
        elif self.mode == 'anomaly' and not self.differential:
            self._test_anomaly()
        else:
            raise ValueError('Something wrong with mode selection')

    @torch.no_grad()
    def _test_class(self):
        all_preds = None
        all_labels = None
        for batch_index, data in enumerate(self.test_loader):
            batch_preds, batch_labels = self.test_step(data, metrics=self.metrics)
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
        # Define classification metrics
        metrics = {'acc': Accuracy(),
                   'f1': F1Score()}
        all_preds = None
        all_labels = None
        for batch_index, data in enumerate(self.test_loader):
            batch_preds, batch_labels = self.test_step(data, metrics, threshold, threshold_metric=threshold_metric)
            # Append batch predictions and labels to the list containing every prediction and every label
            if batch_index == 0:
                all_preds = batch_preds
                all_labels = batch_labels
            else:
                all_preds = torch.cat((all_preds, batch_preds), dim=0)
                all_labels = torch.cat((all_labels, batch_labels), dim=0)
            # Compute metrics over predictions
            scores = {}
            for metric_name, metric_object in metrics.items():
                scores[metric_name] = metric_object.compute(y_pred=all_preds,
                                                            y_true=all_labels)
            self.print_test_message(index_batch=batch_index, metrics=scores)
        print()

    @torch.no_grad()
    def _test_anomaly_differential(self):
        threshold_metric = 'mse'
        # Get threshold from training
        threshold = self.get_threshold(metric=threshold_metric)
        # Define empty list for simulations metrics
        simulations_false_alarms = []
        simulations_true_alarms = []
        simulations_exact_alarms = []
        # Iterate over all simulations belonging to the test set
        for sim_index, simulation in self.test_loader.items():
            # Iterate over each sample of the simulation
            predictions = []
            labels = []
            for sample_index, sample in enumerate(simulation):
                sample = sample.to(self.device)
                prediction, label = self.test_step(sample, metrics=None, threshold=threshold, threshold_metric=threshold_metric)
                # Append prediction and label to the list containing every prediction and label of the simulation
                predictions.append(prediction.numpy().item())
                labels.append(label.numpy().item())
            # print('predictions: {}'.format(predictions))
            # print('labels: {}'.format(labels))
            # Get number of false alarms, true alarms and the behaviour at the
            # starting point of the attack (exact alarm) for the simulation
            false_alarms = 0
            true_alarms = 0
            exact_alarm = 0
            for index in range(len(predictions)):
                if predictions[index] == 1 and labels[index] == 1:
                    true_alarms += 1
                elif predictions[index] == 1 and labels[index] == 0:
                    false_alarms += 1
                else:
                    pass
                if labels[index] == 1 and labels[index - 1] == 0:
                    if predictions[index] == 1 and labels[index] == 1:
                        exact_alarm = 1
            # Append the values to the list of values defining performances over simulations
            simulations_false_alarms.append(false_alarms)
            simulations_true_alarms.append(true_alarms)
            simulations_exact_alarms.append(exact_alarm)
        # Print the results
        print('True alarms over testing simulations: {}'.format(simulations_true_alarms))
        print('False alarms over testing simulations: {}'.format(simulations_false_alarms))
        print('Exact alarm over testing simulations: {}'.format(simulations_exact_alarms))

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
        print('len(values) = {}'.format(len(values)))
        if 0 < self.percentile < 1:
            threshold_index = int(len(sorted_values) * self.percentile)
        elif self.percentile == 1:
            threshold_index = int(len(sorted_values)) - 1
        else:
            raise ValueError('Percentile should be between 0 and 1')
        threshold = sorted_values[threshold_index]
        print('sorted_values: {}'.format(sorted_values))
        print('{} threshold obtained: {}'.format(metric.upper(), threshold))
        return threshold

    @torch.no_grad()
    def test_step(self, data, metrics, threshold=None, threshold_metric='mae'):
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
            # print('preds_mses: {:.5f} -> label: {} -> '
            #       'prediction: {} -> freq: {}'.format(preds_mses,
            #                                           data.attack_is_on.item(),
            #                                           y_pred.item(),
            #                                           data.frequency.item()))
        elif self.mode == 'class':
            y_pred = self.model(data)
        # Get labels from data
        y_true = data.attack_is_on
        # Return predictions and labels
        return y_pred, y_true

    def print_message(self, epoch, index_train_batch, train_loss, train_mets,
                      index_val_batch, val_loss, val_mets):
        message = '| Epoch: {}/{} | LR: {:.5f} |'.format(epoch + 1,
                                                         self.epochs,
                                                         self.lr_scheduler.optimizer.param_groups[0]['lr'])
        bar_length = 10
        total_train_batches = len(self.train_loader)
        progress = float(index_train_batch) / float(total_train_batches)
        if progress >= 1.:
            progress = 1
        block = int(round(bar_length * progress))
        message += '[{}]'.format('=' * block + ' ' * (bar_length - block))
        message += '| TRAIN: loss={:.5f} '.format(train_loss)
        if train_mets is not None:
            train_metrics_message = ''
            for metric_name, metric_value in train_mets.items():
                train_metrics_message += '{}={:.5f} '.format(metric_name,
                                                             metric_value)
            message += train_metrics_message
        # Add validation loss
        if val_mets is not None:
            bar_length = 10
            total_val_batches = len(self.val_loader)
            progress = float(index_val_batch) / float(total_val_batches)
            if progress >= 1.:
                progress = 1
            block = int(round(bar_length * progress))
            message += '|[{}]'.format('=' * block + ' ' * (bar_length - block))
            message += '| VAL: loss={:.5f} '.format(val_loss)
            val_metrics_message = ''
            for metric_name, metric_value in val_mets.items():
                val_metrics_message += '{}={:.5f} '.format(metric_name,
                                                           metric_value)
            message += val_metrics_message
        message += '|'
        # message += 'Loss weights are: {}'.format(self.criterion_reg.weight.numpy())
        print(message, end='\r')

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
