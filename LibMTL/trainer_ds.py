import torch, os
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import deepspeed
from deepspeed.moe.utils import split_params_into_different_moe_groups_for_optimizer
from deepspeed.accelerator import get_accelerator

from LibMTL._record import _PerformanceMeter
from LibMTL.utils import count_parameters
import LibMTL.weighting as weighting_method
import LibMTL.architecture as architecture_method
from LibMTL.model.resnet import resnet50 

def create_moe_param_groups(model):
    """Create separate parameter groups for each expert."""
    parameters = {"params": [p for p in model.parameters()], "name": "parameters"}
    return split_params_into_different_moe_groups_for_optimizer(parameters)

class Trainer_DS(nn.Module):
    r'''A Multi-Task Learning Trainer.

    This is a unified and extensible training framework for multi-task learning. 

    Args:
        task_dict (dict): A dictionary of name-information pairs of type (:class:`str`, :class:`dict`). \
                            The sub-dictionary for each task has four entries whose keywords are named **metrics**, \
                            **metrics_fn**, **loss_fn**, **weight** and each of them corresponds to a :class:`list`.
                            The list of **metrics** has ``m`` strings, repersenting the name of ``m`` metrics \
                            for this task. The list of **metrics_fn** has two elements, i.e., the updating and score \
                            functions, meaning how to update thoes objectives in the training process and obtain the final \
                            scores, respectively. The list of **loss_fn** has ``m`` loss functions corresponding to each \
                            metric. The list of **weight** has ``m`` binary integers corresponding to each \
                            metric, where ``1`` means the higher the score is, the better the performance, \
                            ``0`` means the opposite.                           
        weighting (class): A weighting strategy class based on :class:`LibMTL.weighting.abstract_weighting.AbsWeighting`.
        architecture (class): An architecture class based on :class:`LibMTL.architecture.abstract_arch.AbsArchitecture`.
        encoder_class (class): A neural network class.
        decoders (dict): A dictionary of name-decoder pairs of type (:class:`str`, :class:`torch.nn.Module`).
        rep_grad (bool): If ``True``, the gradient of the representation for each task can be computed.
        multi_input (bool): Is ``True`` if each task has its own input data, otherwise is ``False``. 
        optim_param (dict): A dictionary of configurations for the optimizier.
        scheduler_param (dict): A dictionary of configurations for learning rate scheduler. \
                                 Set it to ``None`` if you do not use a learning rate scheduler.
        kwargs (dict): A dictionary of hyperparameters of weighting and architecture methods.

    .. note::
            It is recommended to use :func:`LibMTL.config.prepare_args` to return the dictionaries of ``optim_param``, \
            ``scheduler_param``, and ``kwargs``.

    Examples::
        
        import torch.nn as nn
        from LibMTL import Trainer
        from LibMTL.loss import CE_loss_fn
        from LibMTL.metrics import acc_update_fun, acc_score_fun
        from LibMTL.weighting import EW
        from LibMTL.architecture import HPS
        from LibMTL.model import ResNet18
        from LibMTL.config import prepare_args

        task_dict = {'A': {'metrics': ['Acc'],
                           'metrics_fn': [acc_update_fun, acc_score_fun],
                           'loss_fn': [CE_loss_fn],
                           'weight': [1]}}
        
        decoders = {'A': nn.Linear(512, 31)}
        
        # You can use command-line arguments and return configurations by ``prepare_args``.
        # kwargs, optim_param, scheduler_param = prepare_args(params)
        optim_param = {'optim': 'adam', 'lr': 1e-3, 'weight_decay': 1e-4}
        scheduler_param = {'scheduler': 'step'}
        kwargs = {'weight_args': {}, 'arch_args': {}}

        trainer = Trainer(task_dict=task_dict,
                          weighting=EW,
                          architecture=HPS,
                          encoder_class=ResNet18,
                          decoders=decoders,
                          rep_grad=False,
                          multi_input=False,
                          optim_param=optim_param,
                          scheduler_param=scheduler_param,
                          **kwargs)

    '''
    def __init__(self, task_dict, weighting, architecture, encoder_class, decoders, 
                 rep_grad, multi_input, optim_param, scheduler_param, 
                 save_path=None, load_path=None, ds_config=None, ds_args=None, **kwargs):
        super(Trainer_DS, self).__init__()
        
        self.device = torch.device('cuda:0')
        self.kwargs = kwargs
        self.task_dict = task_dict
        self.task_num = len(task_dict)
        self.task_name = list(task_dict.keys())
        self.rep_grad = rep_grad
        self.multi_input = multi_input
        self.scheduler_param = scheduler_param
        self.save_path = save_path
        self.load_path = load_path

        self.ds_config = ds_config
        self.ds_args = ds_args

        self._prepare_model(weighting, architecture, encoder_class, decoders)
        #self._prepare_optimizer(optim_param, scheduler_param)
        
        self.meter = _PerformanceMeter(self.task_dict, self.multi_input)
        
    def _prepare_model(self, weighting, architecture, encoder_class, decoders):
        weighting = weighting_method.__dict__[weighting] 
        architecture = architecture_method.__dict__[architecture]
        
        class MTLmodel(architecture, weighting):
            def __init__(self, task_name, encoder_class, decoders, rep_grad, multi_input, device, kwargs):
                super(MTLmodel, self).__init__(task_name, encoder_class, decoders, rep_grad, multi_input, device, **kwargs)
                self.init_param()
                
        self.model = MTLmodel(task_name=self.task_name, 
                              encoder_class=encoder_class, 
                              decoders=decoders, 
                              rep_grad=self.rep_grad, 
                              multi_input=self.multi_input,
                              device=self.device,
                              kwargs=self.kwargs['arch_args']).to(self.device)
        if self.load_path is not None:
            if os.path.isdir(self.load_path):
                self.load_path = os.path.join(self.load_path, 'best.pt')
            self.model.load_state_dict(torch.load(self.load_path), strict=False)
            print('Load Model from - {}'.format(self.load_path))

        count_parameters(self.model)
        # Step 2. Define the network with DeepSpeed.
        # Get list of parameters that require gradients.
        parameters = filter(lambda p: p.requires_grad, self.model.parameters())

    def _prepare_optimizer(self, optim_param, scheduler_param):
        optim_dict = {
                'sgd': torch.optim.SGD,
                'adam': torch.optim.Adam,
                'adagrad': torch.optim.Adagrad,
                'rmsprop': torch.optim.RMSprop,
            }
        scheduler_dict = {
                'exp': torch.optim.lr_scheduler.ExponentialLR,
                'step': torch.optim.lr_scheduler.StepLR,
                'cos': torch.optim.lr_scheduler.CosineAnnealingLR,
                'reduce': torch.optim.lr_scheduler.ReduceLROnPlateau,
            }
        optim_arg = {k: v for k, v in optim_param.items() if k != 'optim'}
        self.optimizer = optim_dict[optim_param['optim']](self.model.parameters(), **optim_arg)
        if scheduler_param is not None:
            scheduler_arg = {k: v for k, v in scheduler_param.items() if k != 'scheduler'}
            self.scheduler = scheduler_dict[scheduler_param['scheduler']](self.optimizer, **scheduler_arg)
        else:
            self.scheduler = None

    def _prepare_optimizer_ds(self, optimizer):
        scheduler_dict = {
                'exp': torch.optim.lr_scheduler.ExponentialLR,
                'step': torch.optim.lr_scheduler.StepLR,
                'cos': torch.optim.lr_scheduler.CosineAnnealingLR,
                'reduce': torch.optim.lr_scheduler.ReduceLROnPlateau,
            }
        if self.scheduler_param is not None:
            scheduler_arg = {k: v for k, v in self.scheduler_param.items() if k != 'scheduler'}
            self.scheduler = scheduler_dict[self.scheduler_param['scheduler']](optimizer, **scheduler_arg)
        else:
            self.scheduler = None

    def _process_data(self, loader):
        try:
            data, label = next(loader[1])
        except:
            loader[1] = iter(loader[0])
            data, label = next(loader[1])
        data = data.to(self.device, non_blocking=True)
        if not self.multi_input:
            for task in self.task_name:
                label[task] = label[task].to(self.device, non_blocking=True)
        else:
            label = label.to(self.device, non_blocking=True)
        return data, label
    
    def process_preds(self, preds, task_name=None):
        r'''The processing of prediction for each task. 

        - The default is no processing. If necessary, you can rewrite this function. 
        - If ``multi_input`` is ``True``, ``task_name`` is valid and ``preds`` with type :class:`torch.Tensor` is the prediction of this task.
        - otherwise, ``task_name`` is invalid and ``preds`` is a :class:`dict` of name-prediction pairs of all tasks.

        Args:
            preds (dict or torch.Tensor): The prediction of ``task_name`` or all tasks.
            task_name (str): The string of task name.
        '''
        return preds

    def _compute_loss(self, preds, gts, task_name=None):
        if not self.multi_input:
            train_losses = torch.zeros(self.task_num).to(self.device)
            for tn, task in enumerate(self.task_name):
                train_losses[tn] = self.meter.losses[task]._update_loss(preds[task], gts[task])
        else:
            train_losses = self.meter.losses[task_name]._update_loss(preds, gts)
        return train_losses
        
    def _prepare_dataloaders(self, dataloaders):
        if not self.multi_input:
            loader = [dataloaders, iter(dataloaders)]
            return loader, len(dataloaders)
        else:
            loader = {}
            batch_num = []
            for task in self.task_name:
                loader[task] = [dataloaders[task], iter(dataloaders[task])]
                batch_num.append(len(dataloaders[task]))
            return loader, batch_num

    def train(self, trainset, testloader, epochs, 
              val_dataset=None, return_weight=False):
        r'''The training process of multi-task learning.

        Args:
            train_dataloaders (dict or torch.utils.data.DataLoader): The dataloaders used for training. \
                            If ``multi_input`` is ``True``, it is a dictionary of name-dataloader pairs. \
                            Otherwise, it is a single dataloader which returns data and a dictionary \
                            of name-label pairs in each iteration.

            test_dataloaders (dict or torch.utils.data.DataLoader): The dataloaders used for the validation or testing. \
                            The same structure with ``train_dataloaders``.
            epochs (int): The total training epochs.
            return_weight (bool): if ``True``, the loss weights will be returned.
        '''
        #train_loader, train_batch = self._prepare_dataloaders(train_dataloaders)
        #train_batch = max(train_batch) if self.multi_input else train_batch
        self.model.train_loss_buffer = np.zeros([self.task_num, epochs])
        self.model.epochs = epochs
        parameters = filter(lambda p: p.requires_grad, self.model.parameters())
        model_engine, optimizer, trainloader, __ = deepspeed.initialize(
            args=self.ds_args,
            model=self.model,
            model_parameters=parameters,
            training_data=trainset,
            config=self.ds_config,
        )
        self._prepare_optimizer_ds(optimizer)
        train_loader, train_batch = self._prepare_dataloaders(trainloader)
        local_device = get_accelerator().device_name(model_engine.local_rank)
        local_rank = model_engine.local_rank
        target_dtype = None
        if model_engine.bfloat16_enabled():
            target_dtype = torch.bfloat16
        elif model_engine.fp16_enabled():
            target_dtype = torch.half

        model_engine.train()
        for epoch in range(epochs):
            self.model.epoch = epoch
            lrs = self.scheduler.get_last_lr()
            for g, lr in zip(model_engine.optimizer.param_groups,lrs):
                g['lr'] = lr
            self.meter.record_time('begin')
            for batch_index in range(train_batch):
                train_inputs, train_gts = self._process_data(train_loader)
                train_inputs = train_inputs.to(local_device)
                #train_gts = train_gts.to(local_device)
                for key in train_gts:
                    train_gts[key] = train_gts[key].to(local_device)
                if target_dtype != None:
                    train_inputs = train_inputs.to(target_dtype)
                train_preds = model_engine(train_inputs)
                train_preds = self.process_preds(train_preds)
                train_losses = self._compute_loss(train_preds, train_gts)
                self.meter.update(train_preds, train_gts)

                # Change this in debugging
                #optimizer.zero_grad(set_to_none=False)
                w = self.model.backward(train_losses, model_engine.backward, **self.kwargs['weight_args'])
                model_engine.step()
            # self.model.epoch = epoch
            # self.model.train()
            # self.meter.record_time('begin')
            # for batch_index in range(train_batch):
            #     if not self.multi_input:
            #         train_inputs, train_gts = self._process_data(train_loader)
            #         train_preds = self.model(train_inputs)
            #         train_preds = self.process_preds(train_preds)
            #         train_losses = self._compute_loss(train_preds, train_gts)
            #         self.meter.update(train_preds, train_gts)
            #     else:
            #         train_losses = torch.zeros(self.task_num).to(self.device)
            #         for tn, task in enumerate(self.task_name):
            #             train_input, train_gt = self._process_data(train_loader[task])
            #             train_pred = self.model(train_input, task)
            #             train_pred = train_pred[task]
            #             train_pred = self.process_preds(train_pred, task)
            #             train_losses[tn] = self._compute_loss(train_pred, train_gt, task)
            #             self.meter.update(train_pred, train_gt, task)

            #     self.optimizer.zero_grad(set_to_none=False)
            #     w = self.model.backward(train_losses, **self.kwargs['weight_args'])
            #     if w is not None:
            #         self.batch_weight[:, epoch, batch_index] = w
            #     self.optimizer.step()
            
            self.meter.record_time('end')
            self.meter.get_score()
            self.model.train_loss_buffer[:, epoch] = self.meter.loss_item
            self.meter.display(epoch=epoch, mode='train')
            self.meter.reinit()
            
            # if val_dataloaders is not None:
            #     self.meter.has_val = True
            #     val_improvement = self.test(val_dataloaders, epoch, mode='val', return_improvement=True)

            self.test(model_engine, local_device, testloader, epoch, mode='test')
            if self.scheduler is not None:
                # if self.scheduler_param['scheduler'] == 'reduce' and val_dataloaders is not None:
                #     self.scheduler.step(val_improvement)
                # else:
                
                #Not aware of it's effects as its outside the DeepSpeed engine
                self.scheduler.step()
                # pass
            if self.save_path is not None and self.meter.best_result['epoch'] == epoch:
                torch.save(self.model.state_dict(), os.path.join(self.save_path, 'best.pt'))
                print('Save Model {} to {}'.format(epoch, os.path.join(self.save_path, 'best.pt')))
        self.meter.display_best_result()

    def test(self, model_engine, local_device, testloaders, epoch=None, mode='test', return_improvement=False):
        r'''The test process of multi-task learning.

        Args:
            test_dataloaders (dict or torch.utils.data.DataLoader): If ``multi_input`` is ``True``, \
                            it is a dictionary of name-dataloader pairs. Otherwise, it is a single \
                            dataloader which returns data and a dictionary of name-label pairs in each iteration.
            epoch (int, default=None): The current epoch. 
        '''
        test_loader, test_batch = self._prepare_dataloaders(testloaders)
        
        model_engine.eval()
        self.meter.record_time('begin')
        target_dtype = None
        if model_engine.bfloat16_enabled():
            target_dtype = torch.bfloat16
        elif model_engine.fp16_enabled():
            target_dtype = torch.half
        with torch.no_grad():
            #if not self.multi_input:
            for batch_index in range(test_batch):
                test_inputs, test_gts = self._process_data(test_loader)
                test_inputs = test_inputs.to(local_device)
                if target_dtype != None:
                    test_inputs = test_inputs.to(target_dtype)
                test_preds = model_engine(test_inputs)
                for key in test_gts:
                    test_gts[key] = test_gts[key].to(local_device)
                test_preds = self.process_preds(test_preds)
                for key in test_gts:
                    test_preds[key] = test_preds[key].to(local_device)
                test_losses = self._compute_loss(test_preds, test_gts)
                self.meter.update(test_preds, test_gts)
            # else:
            #     for tn, task in enumerate(self.task_name):
            #         for batch_index in range(test_batch[tn]):
            #             test_input, test_gt = self._process_data(test_loader[task])
            #             test_pred = self.model(test_input, task)
            #             test_pred = test_pred[task]
            #             test_pred = self.process_preds(test_pred)
            #             test_loss = self._compute_loss(test_pred, test_gt, task)
            #             self.meter.update(test_pred, test_gt, task)
        self.meter.record_time('end')
        self.meter.get_score()
        self.meter.display(epoch=epoch, mode=mode)
        improvement = self.meter.improvement
        self.meter.reinit()
        if return_improvement:
            return improvement
