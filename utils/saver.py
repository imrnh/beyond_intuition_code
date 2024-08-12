import os
import torch
from collections import OrderedDict
import glob


class Saver(object):

    def __init__(self, args):
        self.results_dir = None
        self.args = args
        self.directory = os.path.join('run', args.train_dataset, args.checkname)
        self.runs = sorted(glob.glob(os.path.join(self.directory, 'experiment_*')))
        run_id = int(self.runs[-1].split('_')[-1]) + 1 if self.runs else 0

        self.experiment_dir = os.path.join(self.directory, 'experiment_{}'.format(str(run_id)))
        if not os.path.exists(self.experiment_dir):
            os.makedirs(self.experiment_dir)

    def save_checkpoint(self, state, filename='checkpoint.pth.tar'):
        """Saves checkpoint to disk"""
        filename = os.path.join(self.experiment_dir, filename)
        torch.save(state, filename)

    def save_experiment_config(self):
        logfile = os.path.join(self.experiment_dir, 'parameters.txt')
        log_file = open(logfile, 'w')
        p = OrderedDict()
        p['train_dataset'] = self.args.train_dataset
        p['lr'] = self.args.lr
        p['epoch'] = self.args.epochs

        for key, val in p.items():
            log_file.write(key + ':' + str(val) + '\n')
        log_file.close()

    def create_directories(self):
        self.results_dir = os.path.join(self.experiment_dir, 'results')
        if not os.path.exists(self.results_dir):
            os.makedirs(self.results_dir)
        if not os.path.exists(os.path.join(self.results_dir, 'input')):
            os.makedirs(os.path.join(self.results_dir, 'input'))
        if not os.path.exists(os.path.join(self.results_dir, 'explain')):
            os.makedirs(os.path.join(self.results_dir, 'explain'))

        self.args.exp_img_path = os.path.join(self.results_dir, 'explain/img')
        if not os.path.exists(self.args.exp_img_path):
            os.makedirs(self.args.exp_img_path)
        self.args.exp_np_path = os.path.join(self.results_dir, 'explain/np')
        if not os.path.exists(self.args.exp_np_path):
            os.makedirs(self.args.exp_np_path)
