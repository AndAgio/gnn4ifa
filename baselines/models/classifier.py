import numpy as np
from sklearn import svm, tree


class Classifier:
    def __init__(self, chosen_model='svm', data_mode='avg', feat_set='all', routers=None):
        self.chosen_model = chosen_model
        self.data_mode = data_mode
        self.routers = routers
        if feat_set == 'all':
            self.feat_set = ['pit_size',
                             'drop_rate',
                             'in_interests',
                             'out_interests',
                             'in_data',
                             'out_data',
                             'in_nacks',
                             'out_nacks',
                             'in_satisfied_interests',
                             'in_timedout_interests',
                             'out_satisfied_interests',
                             'out_timedout_interests']
        else:
            self.feat_set = feat_set
        # Define ML model depending on the selected model and the number of features
        if self.chosen_model == 'svm':
            if self.data_mode in ['avg', 'cat']:
                self.model = svm.SVC()
            elif self.data_mode == 'single':
                self.model = {rout: svm.SVC for rout in self.routers}
            else:
                raise ValueError('Data mode {} is not available!'.format(self.data_mode))
        elif self.chosen_model == 'tree':
            if self.data_mode in ['avg', 'cat']:
                self.model = tree.DecisionTreeClassifier()
            elif self.data_mode == 'single':
                self.model = {rout: tree.DecisionTreeClassifier() for rout in self.routers}
            else:
                raise ValueError('Data mode {} is not available!'.format(self.data_mode))
        else:
            raise ValueError('Model {} is not available!'.format(self.chosen_model))

    def fit(self, dataset):
        features, labels = self.extract_features_and_labels(dataset)
        if self.data_mode in ['avg', 'cat']:
            self.model.fit(features, labels)
        elif self.data_mode in ['single']:
            raise ValueError('Fitting for model {} and data mode {} is not available yet!'.format(self.chosen_model,
                                                                                                  self.data_mode))
        else:
            raise ValueError('Data mode {} is not available!'.format(self.data_mode))

    def test(self, dataset, metrics, verbose=True):
        features, labels = self.extract_features_and_labels(dataset)
        if self.data_mode in ['avg', 'cat']:
            predictions = self.model.predict(features)
            print('predictions: {}'.format(predictions))
            # Compute metrics over predictions
            scores = {}
            for metric_name, metric_object in metrics.items():
                scores[metric_name] = metric_object.compute(y_pred=predictions,
                                                            y_true=labels)
                if verbose:
                    print('Metrics {}={:.3f}'.format(metric_name, scores[metric_name]*100))
        elif self.data_mode in ['single']:
            raise ValueError('Fitting for model {} and data mode {} is not available yet!'.format(self.chosen_model,
                                                                                                  self.data_mode))
        else:
            raise ValueError('Data mode {} is not available!'.format(self.data_mode))

    def concat_features(self, row):
        return [row[feat] for feat in self.feat_set]

    def extract_features_and_labels(self, dataset):
        if self.data_mode == 'avg':
            for feat in self.feat_set:
                print('dataset[feat]: {}'.format(dataset[feat]))
                dataset[feat] = (dataset[feat] - dataset[feat].min()) / \
                                      (dataset[feat].max() - dataset[feat].min())
                # dataset[feat] = (dataset[feat] - dataset[feat].mean()) / dataset[feat].std()
                dataset = dataset.fillna(0)
                print('dataset[feat]: {}'.format(dataset[feat]))
            dataset['feat_vector'] = dataset.apply(lambda row: self.concat_features(row), axis=1)
            features = [np.asarray(feat_vector) for feat_vector in dataset['feat_vector']]
            labels = [np.asarray(label, dtype=np.int8) for label in dataset['attack_is_on']]
        elif self.data_mode == 'cat':
            # Normalise single features using numpy matrix
            # print('dataset[\'feat_vector\'].to_list()[0][:10]: {}'.format(dataset['feat_vector'].to_list()[0][:10]))
            features = np.array(dataset['feat_vector'].to_list())
            # print('features shape: {}'.format(features.shape))
            normalised_features = (features - features.min(0)) / features.ptp(0)
            normalised_features = np.nan_to_num(normalised_features, nan=0)
            # print('normalised_features.shape: {}'.format(normalised_features.shape))
            features = [feat_vector for feat_vector in normalised_features]
            # print('length of features: {}'.format(len(features)))
            # print('features[0][:10]: {}'.format(features[0][:10]))
            labels = [np.asarray(label, dtype=np.int8) for label in dataset['attack_is_on']]
        elif self.data_mode in ['single']:
            raise ValueError('Fitting for model {} and data mode {} is not available yet!'.format(self.chosen_model,
                                                                                                  self.data_mode))
        else:
            raise ValueError('Data mode {} is not available!'.format(self.data_mode))
        return features, labels