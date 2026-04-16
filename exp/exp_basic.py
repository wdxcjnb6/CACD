"""
Base Experiment Class
Provides common infrastructure for all experiment types
"""
import os
import torch


class Exp_Basic(object):
    """
    Base class for all experiments
    
    Subclasses must implement:
        - _build_model(): return the model instance
        - _get_data(flag): return dataset and dataloader
        - train(setting): training loop
        - vali(vali_data, vali_loader, criterion): validation loop
        - test(setting, **kwargs): testing loop
    """
    
    def __init__(self, args):
        """
        Initialize experiment
        
        Args:
            args: argument namespace containing:
                - use_gpu: whether to use GPU
                - gpu: GPU device ID
                - use_multi_gpu: whether to use multiple GPUs
                - devices: comma-separated GPU device IDs (for multi-GPU)
        """
        self.args = args
        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)

    def _build_model(self):
        """
        Build and return the model instance
        Must be implemented by subclasses
        
        Returns:
            nn.Module: model instance
        """
        raise NotImplementedError("Subclass must implement _build_model()")

    def _acquire_device(self):
        """
        Configure computing device (CPU or GPU)
        
        Returns:
            torch.device: configured device
        """
        if self.args.use_gpu:
            # Set visible GPU devices
            os.environ["CUDA_VISIBLE_DEVICES"] = (
                str(self.args.gpu) 
                if not self.args.use_multi_gpu 
                else self.args.devices
            )
            device = torch.device('cuda:0')
            print(f'Using GPU: cuda:{self.args.gpu}')
        else:
            device = torch.device('cpu')
            print('Using CPU')
        return device

    def _get_data(self, flag):
        """
        Get dataset and dataloader
        Must be implemented by subclasses
        
        Args:
            flag: data split flag ('train', 'val', 'test', 'veri')
            
        Returns:
            tuple: (dataset, dataloader)
        """
        raise NotImplementedError("Subclass must implement _get_data()")

    def vali(self, vali_data, vali_loader, criterion):
        """
        Validation loop
        Must be implemented by subclasses
        
        Args:
            vali_data: validation dataset
            vali_loader: validation dataloader
            criterion: loss function
            
        Returns:
            float: validation loss
        """
        raise NotImplementedError("Subclass must implement vali()")

    def train(self, setting):
        """
        Training loop
        Must be implemented by subclasses
        
        Args:
            setting: experiment setting string
            
        Returns:
            model: trained model
        """
        raise NotImplementedError("Subclass must implement train()")

    def test(self, setting, **kwargs):
        """
        Testing loop
        Must be implemented by subclasses
        
        Args:
            setting: experiment setting string
            **kwargs: additional test arguments
            
        Returns:
            results: test results (format depends on subclass)
        """
        raise NotImplementedError("Subclass must implement test()")