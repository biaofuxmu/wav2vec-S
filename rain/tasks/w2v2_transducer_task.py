import logging
from argparse import Namespace
import numpy as np
import torch
from dataclasses import dataclass

from fairseq.tasks import register_task
from fairseq.optim import FP16Optimizer

from . import w2v2_s2s_task

logger = logging.getLogger(__name__)


@register_task("w2v2_transducer", dataclass= w2v2_s2s_task.W2V2SimulSTTaskConfig)
class W2V2TransducerTask(w2v2_s2s_task.W2V2SimulSTTask):
    def inference_step(
        self, generator, models, sample, prefix_tokens=None, constraints=None
    ):
        raise NotImplementedError("TODO")

    def build_generator(self, models, args, seq_gen_cls = None, extra_gen_cls_kwargs=None):
        raise NotImplementedError("TODO")
    
    def train_step(
        self, sample, model, criterion, optimizer, update_num, ignore_grad=False
    ):
        """
            model.train_step returns
            {"loss":losses[0], "loss_prob":losses[1], "loss_delay":losses[2], "sample_size": B}
        """
        model.train()
        model.set_num_updates(update_num)
        scaler=None
        if hasattr(optimizer, "scaler"):
            scaler= optimizer.scaler
        if ignore_grad:
            loss_info = model.eval_step(sample)
        else:
            loss_info= model.train_step(sample,scaler=scaler)
            if isinstance(optimizer, FP16Optimizer):
                optimizer._needs_sync=True
        loss, sample_size, logging_output = criterion(loss_info, sample, model)
        return loss,sample_size, logging_output
    
    def valid_step(self, sample, model, criterion):
        model.eval()
        with torch.no_grad():
            loss_info = model.eval_step(sample)
        loss, sample_size, logging_output = criterion(loss_info, sample, model)
        return loss,sample_size, logging_output
    
    def build_criterion(self, args: Namespace):
        """
        should only build fake criterion
        """
        from rain.criterions.fake_creterion import FakeCriterion

        return FakeCriterion(self)