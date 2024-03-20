import torch


class Early_stopper(object):
    def __init__(self, model_file, mode='higher', patience=70, tolerance=0.0):
        self.model_file = model_file
        assert mode in ['higher', 'lower']
        self.mode = mode
        if self.mode == 'higher':
            self._check = self._check_higher
        else:
            self._check = self._check_lower
        self.patience = patience
        self.tolerance = tolerance
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def _check_higher(self, score, prev_best_score):
        # return (score > prev_best_score)
        return score / prev_best_score > 1 + self.tolerance

    def _check_lower(self, score, prev_best_score):
        # return (score < prev_best_score)
        return prev_best_score / score > 1 + self.tolerance

    def load_model(self, model_obj, my_device, strict=False, mine=False):
        '''Load model saved with early stopping.'''
        if not mine:
            model_obj.load_state_dict(torch.load(self.model_file, map_location=my_device)['model_state_dict'], strict=strict)
        else:
            params = torch.load(self.model_file, map_location=my_device)['model']
            params_ = {}
            for k, v in params.items():
                k_ = 'module.' + k[6:]
                params_[k_] = v 
            del params
            model_obj.load_state_dict(params_, strict=strict)

    def save_model(self, model_obj):
        '''Saves model when the metric on the validation set gets improved.'''
        torch.save({'model_state_dict': model_obj.state_dict()}, self.model_file)

    def step(self, score, model_obj):
        if self.best_score is None:
            self.best_score = score
            self.save_model(model_obj)
        elif self._check(score, self.best_score):
            self.best_score = score
            self.save_model(model_obj)
            self.counter = 0
        else:
            self.counter += 1
            print(f'# EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        print(f'# Current best performance {float(self.best_score):.3f}')
        return self.early_stop
