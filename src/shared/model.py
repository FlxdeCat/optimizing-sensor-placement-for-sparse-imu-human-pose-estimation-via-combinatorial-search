"""Configurable adapter around the pinned upstream recurrent architecture."""
from pathlib import Path
import torch
from torch import nn
from config import PROJECT,ModelSpec
from .vendor_adapter import _load_module,verify_vendor

_CLASS=None


def parameter_count(model):return sum(p.numel() for p in model.parameters())


class PoseModel(nn.Module):
    def __init__(self,spec):
        super().__init__()
        global _CLASS
        if _CLASS is None:
            root=PROJECT/"third_party"/"imuoptimize";verify_vendor(root)
            _CLASS=_load_module("_simple_imuopt_birnn",root/"src/model/full_ds/birnn.py").StepOneRNN
        self.spec=spec
        self.backbone=_CLASS(num_hidden=spec.rnn_hidden)
        if spec.rnn_layers!=2 or not spec.rnn_bidirectional or spec.rnn_dropout!=0:
            self.backbone.lstm=nn.LSTM(spec.rnn_hidden,spec.rnn_hidden,spec.rnn_layers,
                bidirectional=spec.rnn_bidirectional,dropout=spec.rnn_dropout,batch_first=True)
            self.backbone.linear2=nn.Linear(spec.rnn_hidden*(2 if spec.rnn_bidirectional else 1),216)

    def forward(self,x,valid=None):
        if x.ndim!=3 or x.shape[-1]!=288 or not 1<=x.shape[1]<=self.spec.max_window_length:raise ValueError("Model input must be [B,T,288] within the configured window limit.")
        if valid is None:return self.backbone(x).reshape(x.shape[0],x.shape[1],24,3,3)
        if valid.shape!=x.shape[:2] or valid.dtype!=torch.bool:raise ValueError("valid must be Boolean [B,T].")
        lengths=valid.sum(1)
        expected=torch.arange(x.shape[1],device=x.device)[None,:]<lengths[:,None]
        if bool((lengths<1).any()) or not torch.equal(expected,valid):raise ValueError("Valid frames must be a nonempty contiguous prefix.")
        if bool((lengths==x.shape[1]).all()):return self.forward(x)
        result=x.new_zeros((x.shape[0],x.shape[1],24,3,3))
        for length in lengths.unique().tolist():
            indices=(lengths==length).nonzero(as_tuple=True)[0]
            pred=self.forward(x.index_select(0,indices)[:,:length])
            padded=torch.cat((pred,pred.new_zeros((len(indices),x.shape[1]-length,24,3,3))),dim=1)
            result=result.index_copy(0,indices,padded.to(result.dtype))
        return result


def build_model(architecture,spec=ModelSpec()):
    if architecture!="birnn":raise ValueError("Only BiRNN is implemented.")
    return PoseModel(spec)
