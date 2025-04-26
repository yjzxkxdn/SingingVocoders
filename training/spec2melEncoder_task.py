import pathlib
import random

import numpy as np
import torch.nn.functional as F
import torch.utils.data
import torchaudio
from matplotlib import pyplot as plt
from torch import nn
from torch.utils.data import Dataset
import lightning.pytorch as pl
from lightning.pytorch.utilities.rank_zero import rank_zero_debug, rank_zero_info, rank_zero_only
from torchmetrics import Metric, MeanMetric

from models.specHiFivae.models import Encoder, AttrDict
import utils
from utils.wav2mel import PitchAdjustableMelSpectrogram
from utils.training_utils import get_latest_checkpoint_path


def spec_to_figure(spec, vmin=None, vmax=None):
    if isinstance(spec, torch.Tensor):
        spec = spec.cpu().numpy()
    fig = plt.figure(figsize=(12, 9), dpi=100)
    plt.pcolor(spec.T, vmin=vmin, vmax=vmax)
    plt.tight_layout()
    return fig


def dynamic_range_compression_torch(x, C=1, clip_val=1e-9):
    return torch.log(torch.clamp(x, min=clip_val) * C)


def wav_aug(wav, hop_size, speed=1):
    orig_freq = int(np.round(hop_size * speed))
    new_freq = hop_size
    resample = torchaudio.transforms.Resample(
        orig_freq=orig_freq,
        new_freq=new_freq,
        lowpass_filter_width=128
    )
    wav_resampled = resample(wav)
    del resample
    return wav_resampled

class Spec2MelEncoder_dataset(Dataset):

    def __init__(self, config: dict, data_dir, infer=False):
        super().__init__()
        self.config = config

        self.data_dir = data_dir if isinstance(data_dir, pathlib.Path) else pathlib.Path(data_dir)
        with open(self.data_dir, 'r', encoding='utf8') as f:
            fills = f.read().strip().split('\n')
        self.data_index = fills
        self.infer = infer
        self.volume_aug = self.config['volume_aug']
        self.volume_aug_prob = self.config['volume_aug_prob'] if not infer else 0

    def __getitem__(self, index):
        sample = self.get_data(index)
        return sample

    def __len__(self):
        return len(self.data_index)

    def get_data(self, index):
        data_path = pathlib.Path(self.data_index[index])
        data = np.load(data_path)
        return {'linear_spec': data['linear_spec'], 'mel_spec': data['mel']}

    def collater(self, minibatch):
        if self.infer:
            crop_mel_frames = 0
        else:
            crop_mel_frames = self.config['crop_mel_frames']

        for record in minibatch:

            # Filter out records that aren't long enough.
            if record['linear_spec'].shape[0] < crop_mel_frames:
                del record['linear_spec']
                del record['mel_spec']
                continue
            elif record['linear_spec'].shape[0] == crop_mel_frames:
                start = 0
            else:
                start = random.randint(0, record['linear_spec'].shape[0] - 1 - crop_mel_frames)
            end = start + crop_mel_frames
            if self.infer:
                record['linear_spec'] = record['linear_spec'].T
                record['mel_spec'] = record['mel_spec'].T
            else:
                record['linear_spec'] = record['linear_spec'][start:end].T
                record['mel_spec'] = record['mel_spec'][start:end].T

        if self.volume_aug:
            for record in minibatch:
                linear_spec = record['linear_spec']
                mel_spec = record['mel_spec']

                if random.random() < self.volume_aug_prob:
                    max_shift = min(3, -np.max(linear_spec)+5)
                    #print(f"-np.max(linear_spec): {-np.max(linear_spec)+5}")
                    log_mel_shift = random.uniform(-3, max_shift)

                    linear_spec += log_mel_shift
                    mel_spec += log_mel_shift
                
                mel_spec = torch.clamp(torch.from_numpy(mel_spec), min=np.log(1e-5)).numpy()
                linear_spec = torch.clamp(torch.from_numpy(linear_spec), min=np.log(1e-5)).numpy()

                record['linear_spec'] = linear_spec
                record['mel_spec'] = mel_spec

        linear_spec = np.stack([record['linear_spec'] for record in minibatch if 'linear_spec' in record])
        mel_spec = np.stack([record['mel_spec'] for record in minibatch if'mel_spec' in record])


        return {
            'mel_spec': torch.from_numpy(mel_spec), 
            'linear_spec': torch.from_numpy(linear_spec)
        }


class Spec2MelEncoder(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.dataset_cls = None
        self.config = config

        self.clip_grad_norm = self.config['clip_grad_norm']

        self.training_sampler = None
        self.model = None
        self.encoder = None

        self.skip_immediate_validation = False
        self.skip_immediate_ckpt_save = False

        self.valid_losses = {
            'total_loss': MeanMetric()
        }
        self.valid_metric_names = set()
        self.mix_loss = None

        self.automatic_optimization = False
        self.skip_immediate_validations = 0

        self.aux_step = self.config.get('aux_step')
        self.train_dataset = None
        self.valid_dataset = None

    @rank_zero_only
    def print_arch(self):
        utils.print_arch(self)
        
    def register_metric(self, name: str, metric: Metric):
        assert isinstance(metric, Metric)
        setattr(self, name, metric)
        self.valid_metric_names.add(name)
        
    def setup(self, stage):
        self.model = self.build_model()
        self.print_arch()
        self.build_losses_and_metrics()
        self.build_dataset()
        
    def build_dataset(self):
        self.train_dataset = Spec2MelEncoder_dataset(config=self.config,
                                                 data_dir=pathlib.Path(self.config['DataIndexPath']) / self.config[
                                                     'train_set_name'])
        self.valid_dataset = Spec2MelEncoder_dataset(config=self.config,
                                                 data_dir=pathlib.Path(self.config['DataIndexPath']) / self.config[
                                                     'valid_set_name'], infer=True)

    def build_model(self):
        cfg = self.config['model_args']
        h = AttrDict(cfg)
        self.encoder = Encoder(h)

    def build_losses_and_metrics(self):
        self.l1_loss = nn.L1Loss()
        def kl_loss(m, logs):
            return torch.mean(0.5 * (m**2 + torch.exp(logs) - logs - 1).sum(dim=1))
        self.klloss = kl_loss

    def _training_step(self, sample, batch_idx):
        log_dict = {}
        opt = self.optimizers()
        
        linear_spec = sample['linear_spec']
        mel_target = sample['mel_spec']
        
        z, m, logs = self.encoder(linear_spec)
        
        # 重建损失 (encoder输出的均值m与目标mel对齐)
        l1_loss = self.l1_loss(m, mel_target)
        exp_m = torch.exp(m)
        exp_mel_target = torch.exp(mel_target)
        converge_term = torch.mean(torch.linalg.norm(exp_mel_target - exp_m, dim = (1, 2)) / torch.linalg.norm(exp_mel_target + exp_m, dim = (1, 2)))
        
        kl_loss = self.config["kl_weight"] * self.klloss(m, logs)
        total_loss = l1_loss + kl_loss + converge_term
        
        opt.zero_grad()
        self.manual_backward(total_loss)
        opt.step()

        log_dict.update({
            'train/l1_loss': l1_loss.detach(),
            'train/kl_loss': kl_loss.detach(),
            'train/converge_term': converge_term.detach(),
            'train/total_loss': total_loss.detach()
        })
        
        return log_dict
    
    def training_step(self, sample, batch_idx, ):  # todo
        log_outputs = self._training_step(sample, batch_idx)

        # logs to progress bar
        self.log_dict({'loss':sum(log_outputs.values())}, prog_bar=True, logger=False, on_step=True, on_epoch=False)
        if self.global_step % self.config['log_interval'] == 0:
            tb_log = {f'training/{k}': v for k, v in log_outputs.items()}

            self.logger.log_metrics(tb_log, step=self.global_step)

    def _validation_step(self, sample, batch_idx):

        with torch.no_grad():
            linear_spec = sample['linear_spec']  # 输入线性谱
            mel_target = sample['mel_spec']      # 目标mel谱
            
            # 前向传播
            z, m, logs = self.encoder(linear_spec)
            
            l1_loss = self.l1_loss(m, mel_target)
            exp_m = torch.exp(m)
            exp_mel_target = torch.exp(mel_target)
            converge_term = torch.mean(torch.linalg.norm(exp_mel_target - exp_m, dim = (1, 2)) / torch.linalg.norm(exp_mel_target + exp_m, dim = (1, 2)))
            
            kl_loss = self.config["kl_weight"] * self.klloss(m, logs)
            total_loss = l1_loss + kl_loss + converge_term
            self.plot_mel(batch_idx, mel_target.transpose(1,2), m.transpose(1,2), name=f'Encoder_m_{batch_idx}')
            self.plot_mel(batch_idx, mel_target.transpose(1,2), z.transpose(1,2), name=f'Encoder_z_{batch_idx}')
            # 记录损失
            log_dict = {
                'valid/l1_loss': l1_loss.detach(),
                'valid/kl_loss': kl_loss.detach(),
                'valid/converge_term': converge_term.detach(),
                'valid/total_loss': total_loss.detach()
            }
            
        return log_dict, 1
    
    def validation_step(self, sample, batch_idx):
        """

        :param sample:
        :param batch_idx:

        """

        if self.skip_immediate_validation:
            rank_zero_debug(f"Skip validation {batch_idx}")
            return {}
        with torch.autocast(self.device.type, enabled=False):
            losses, weight = self._validation_step(sample, batch_idx)
        losses = {
            'total_loss': sum(losses.values()),
            **losses
        }
        for k, v in losses.items():
            if k not in self.valid_losses:
                self.valid_losses[k] = MeanMetric().to(self.device)
            self.valid_losses[k].update(v, weight=weight)  # weight=1
        return losses

    def plot_mel(self, batch_idx, spec, spec_out, name=None):
        name = f'mel_{batch_idx}' if name is None else name
        vmin = self.config['mel_vmin']
        vmax = self.config['mel_vmax']
        spec_cat = torch.cat([(spec_out - spec).abs() + vmin, spec, spec_out], -1)
        self.logger.experiment.add_figure(name, spec_to_figure(spec_cat[0], vmin, vmax), self.global_step)
        
    def on_validation_epoch_end(self):
        if self.skip_immediate_validation:
            self.skip_immediate_validation = False
            self.skip_immediate_ckpt_save = True
            return
        loss_vals = {k: v.compute() for k, v in self.valid_losses.items()}
        self.log('val_loss', loss_vals['total_loss'], on_epoch=True, prog_bar=True, logger=False, sync_dist=True)
        self.logger.log_metrics({f'validation/{k}': v for k, v in loss_vals.items()}, step=self.global_step)
        for metric in self.valid_losses.values():
            metric.reset()
        metric_vals = {k: getattr(self, k).compute() for k in self.valid_metric_names}
        self.logger.log_metrics({f'metrics/{k}': v for k, v in metric_vals.items()}, step=self.global_step)
        for metric_name in self.valid_metric_names:
            getattr(self, metric_name).reset()

    # noinspection PyMethodMayBeStatic
    def build_scheduler(self, optimizer):
        from utils import build_lr_scheduler_from_config

        scheduler_args = self.config['lr_scheduler_args']
        assert scheduler_args['scheduler_cls'] != ''
        scheduler = build_lr_scheduler_from_config(optimizer, scheduler_args)
        return scheduler

    # noinspection PyMethodMayBeStatic
    def build_optimizer(self, model, optimizer_args):
        from utils import build_object_from_class_name
        print(optimizer_args)

        assert optimizer_args['optimizer_cls'] != ''
        if 'beta1' in optimizer_args and 'beta2' in optimizer_args and 'betas' not in optimizer_args:
            optimizer_args['betas'] = (optimizer_args['beta1'], optimizer_args['beta2'])

        if isinstance(model, nn.ModuleList):
            parameterslist = []
            for i in model:
                parameterslist = parameterslist + list(i.parameters())
            optimizer = build_object_from_class_name(
                optimizer_args['optimizer_cls'],
                torch.optim.Optimizer,
                parameterslist,
                **optimizer_args
            )
        elif isinstance(model, nn.ModuleDict):
            parameterslist = []
            for i in model:
                # parameterslist = parameterslist + list(model[i].parameters())
                parameterslist.append({'params': model[i].parameters()})
            optimizer = build_object_from_class_name(
                optimizer_args['optimizer_cls'],
                torch.optim.Optimizer,
                parameterslist,
                **optimizer_args
            )
        elif isinstance(model, nn.Module):

            optimizer = build_object_from_class_name(
                optimizer_args['optimizer_cls'],
                torch.optim.Optimizer,
                model.parameters(),
                **optimizer_args
            )
        else:
            raise RuntimeError("")

        return optimizer

    def configure_optimizers(self):
        opt = self.build_optimizer(self.encoder, optimizer_args=self.config['optimizer_args'])
        return opt

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_dataset,
                                           collate_fn=self.train_dataset.collater,
                                           batch_size=self.config['batch_size'],
                                           # batch_sampler=self.training_sampler,
                                           num_workers=self.config['ds_workers'],
                                           prefetch_factor=self.config['dataloader_prefetch_factor'],
                                           pin_memory=True,
                                           persistent_workers=True)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.valid_dataset,
                                           collate_fn=self.valid_dataset.collater,
                                           batch_size=1,
                                           # batch_sampler=sampler,
                                           num_workers=self.config['ds_workers'],
                                           prefetch_factor=self.config['dataloader_prefetch_factor'],
                                           shuffle=False)

    def test_dataloader(self):
        return self.val_dataloader()

    def on_test_start(self):
        self.on_validation_start()
        
    def on_validation_start(self):
        self._on_validation_start()
        for metric in self.valid_losses.values():
            metric.to(self.device)
            metric.reset()
            
    def _on_validation_start(self):
        pass

    def test_step(self, sample, batch_idx):
        return self.validation_step(sample, batch_idx)

    def on_test_end(self):
        return self.on_validation_end()

    def on_save_checkpoint(self, checkpoint):
        pass
        # checkpoint['trainer_stage'] = self.trainer.state.stage.value
