import numpy as np

class SimplifiedBaggingRegressor:
    def __init__(self, num_bags, oob=False):
        self.num_bags = num_bags
        self.oob = oob

    def _generate_splits(self, data: np.ndarray):
        '''
        Generate indices for every bag and store in self.indices_list list
        '''
        self.indices_list = []
        self.oob_indices_list = []
        data_length = len(data)
        for _ in range(self.num_bags):
            indices = np.random.choice(data_length, size=data_length, replace=True)
            self.indices_list.append(indices)

            if self.oob:
                mask = np.ones(data_length, dtype=bool)
                mask[indices] = False
                oob_indices = np.where(mask)[0]
                self.oob_indices_list.append(oob_indices)
        
    def fit(self, model_constructor, data, target):
        '''
        Fit model on every bag.
        Model constructor with no parameters (and with no ()) is passed to this function.

        example:

        bagging_regressor = SimplifiedBaggingRegressor(num_bags=10, oob=True)
        bagging_regressor.fit(LinearRegression, X, y)
        '''
        self._generate_splits(data)
        assert len(set(map(len, self.indices_list))) == 1, 'All bags should be of the same length!'
        assert len(self.indices_list[0]) == len(data), 'All bags should contain `len(data)` number of elements!'

        self.models_list = []
        self.data = data
        self.target = target

        for bag in range(self.num_bags):
            model = model_constructor()
            data_bag = data[self.indices_list[bag]]
            target_bag = target[self.indices_list[bag]]
            model.fit(data_bag, target_bag)
            self.models_list.append(model)
        
    def predict(self, data):
        '''
        Get average prediction for every object from passed dataset
        '''
        predictions = [model.predict(data) for model in self.models_list]
        return np.mean(predictions, axis=0)
    
    def _get_oob_predictions_from_every_model(self):
        '''
        Generates list of lists, where list i contains predictions for self.data[i] object
        from all models, which have not seen this object during training phase
        '''
        n = len(self.data)
        list_of_predictions_lists = [[] for _ in range(n)]

        for model, oob_indices in zip(self.models_list, self.oob_indices_list):
            if len(oob_indices) == 0:
                continue
            preds = model.predict(self.data[oob_indices])
            for idx, pred in zip(oob_indices, preds):
                list_of_predictions_lists[idx].append(pred)

        self.list_of_predictions_lists = np.array(list_of_predictions_lists, dtype=object)
    
    def _get_averaged_oob_predictions(self):
        '''
        Compute average prediction for every object from training set.
        If object has been used in all bags on training phase, return None instead of prediction
        '''
        self._get_oob_predictions_from_every_model()
        oob_preds = []

        for preds in self.list_of_predictions_lists:
            if len(preds) == 0:
                oob_preds.append(None)
            else:
                oob_preds.append(np.mean(preds))

        self.oob_predictions = np.array(oob_preds, dtype=object)
        
        
    def OOB_score(self):
        '''
        Compute mean square error for all objects, which have at least one prediction
        '''
        self._get_averaged_oob_predictions()
        true_vals = []
        pred_vals = []

        for i, pred in enumerate(self.oob_predictions):
            if pred is not None:
                true_vals.append(self.target[i])
                pred_vals.append(pred)

        if not true_vals:
            return None
        true_vals = np.array(true_vals)
        pred_vals = np.array(pred_vals)
        return np.mean((true_vals - pred_vals) ** 2)