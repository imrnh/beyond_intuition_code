import os, json
import torch
from typing import List
from monitor_ops import monitor_condition

class Callback:
    def __init__(self,  log_fp, save_dir, save_weight_only, verbose, early_stop, tolerated_diff, patience):
        self.train_log = dict()
        self.log_fp = log_fp
        self.verbose = verbose

        self.save_dir = save_dir
        self.save_weight_only = save_weight_only

        self.early_stop = early_stop
        self.tolerated_diff = tolerated_diff
        self.patience = patience

        self.min_validation_loss = float(1e10)
        self.max_validation_accuracy = float(-1e10)

        os.makedirs(self.save_dir, exist_ok=True)

    def write_file(self, fp, file_content, is_append=True):
        mode = "a" if is_append else "w" 

        with open(fp, mode) as file:
            file.write(file_content)


    # Save model into given path.
    def save_model(self, model, model_name):
        model_path = os.path.join(self.save_dir, f"{model_name}.pth")
        save_objective = model.state_dict() if self.save_weight_only else model
        torch.save(save_objective, model_path)


    """
        monitor are passed as : [validation_loss, "lt"]
        That means, if current_validation_loss (i.e. cmont) is less than previous_validation_loss (i.e. pmont) 
            then return true else false.
    """
    def monitor_condition(self, m, info , i):
        p, c = info[-1][m[i][0]], info[0][m[i][0]]
        return c < p if m[i][1] == "lt" else c > p

    def is_min_loss(self, metrics):
        c = metrics['validation_loss']
        if c < self.min_validation_loss:
            self.min_validation_loss = c
            return True
        return False

    def is_max_acc(self, metrics):
        c = metrics['validation_accuracy']
        if c > self.max_validation_accuracy:
            self.max_validation_accuracy = c
            return True
        return False


    """
        Checkpoint functions.
    """
    def model_checkpoint(self, model, model_prefix, metrics):
        if self.is_min_loss(metrics):
            model_name = model_prefix + "min_validation_loss" + ".pth"
            self.save_model(model, model_prefix)

        if self.is_max_acc(metrics):
            model_name = model_prefix + "max_validation_accuracy" + ".pth"
            self.save_model(model, model_prefix)


    def log_checkpoint(self, metrics, epoch):
        metrics_str = json.dumps(metrics, indent=4)
        self.write_file(self.log_fp, metrics_str, is_append=True) 
        self.train_log[epoch] = metrics
        if self.verbose: print(metrics);


    def __call__(self, model, model_prefix, metrics, epoch):
        self.log_checkpoint(metrics, epoch)
        self.model_checkpoint(model, model_prefix, metrics)