"""
DEIM: DETR with Improved Matching for Fast Convergence
Copyright (c) 2024 The DEIM Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from D-FINE (https://github.com/Peterande/D-FINE)
Copyright (c) 2024 D-FINE authors. All Rights Reserved.
"""

import time
import json
import datetime

import torch

from ..misc import dist_utils, stats

from ._solver import BaseSolver
from .det_engine import train_one_epoch, evaluate
from ..optim.lr_scheduler import FlatCosineLRScheduler


class DetSolver(BaseSolver):

    def fit(self, ):
        self.train()
        args = self.cfg

        n_parameters, model_stats = stats(self.cfg)
        print(model_stats)
        print("-"*42 + "Start training" + "-"*43)

        for i, (name, param) in enumerate(self.model.named_parameters()):
            if i in [194, 195]:
                print(f"Index {i}: {name} - requires_grad: {param.requires_grad}")

        self.self_lr_scheduler = False
        if args.lrsheduler is not None:
            iter_per_epoch = len(self.train_dataloader)
            print("     ## Using Self-defined Scheduler-{} ## ".format(args.lrsheduler))
            self.lr_scheduler = FlatCosineLRScheduler(self.optimizer, args.lr_gamma, iter_per_epoch, total_epochs=args.epoches, 
                                                warmup_iter=args.warmup_iter, flat_epochs=args.flat_epoch, no_aug_epochs=args.no_aug_epoch)
            self.self_lr_scheduler = True
        n_parameters = sum([p.numel() for p in self.model.parameters() if p.requires_grad])
        print(f'number of trainable parameters: {n_parameters}')

        n_parameters = sum([p.numel() for p in self.model.parameters() if not p.requires_grad])
        print(f'number of non-trainable parameters: {n_parameters}')

        top1s = {}
        best_stats = {}
        # evaluate again before resume training
        if self.last_epoch >= 0:
            module = self.ema.module if self.ema else self.model
            test_stats, coco_evaluator = evaluate(
                module,
                self.criterion,
                self.postprocessor,
                self.val_dataloader,
                self.evaluator,
                self.device
            )
            for k in test_stats:
                best_stats[k] = {'epoch': self.last_epoch, 'AP': test_stats[k][0]}
                top1s[k] = test_stats[k][0]
            print(f'best_stat: {best_stats}')

        best_stat_prints = best_stats.copy()
        start_time = time.time()
        start_epoch = self.last_epoch + 1
        for epoch in range(start_epoch, args.epoches):

            self.train_dataloader.set_epoch(epoch)
            # self.train_dataloader.dataset.set_epoch(epoch)
            if dist_utils.is_dist_available_and_initialized():
                self.train_dataloader.sampler.set_epoch(epoch)

            if epoch == self.train_dataloader.collate_fn.stop_epoch:
                if dist_utils.is_dist_available_and_initialized():
                    torch.distributed.barrier()
                self.load_resume_state(str(self.output_dir / (self.ema.testkey + '_best_stg1.pth')))
                self.ema.decay = self.train_dataloader.collate_fn.ema_restart_decay
                print(f'Refresh EMA at epoch {epoch} with decay {self.ema.decay}')

            train_stats = train_one_epoch(
                self.self_lr_scheduler,
                self.lr_scheduler,
                self.model, 
                self.criterion, 
                self.train_dataloader, 
                self.optimizer, 
                self.device, 
                epoch, 
                max_norm=args.clip_max_norm, 
                print_freq=args.print_freq, 
                ema=self.ema, 
                scaler=self.scaler, 
                lr_warmup_scheduler=self.lr_warmup_scheduler,
                writer=self.writer
            )

            if not self.self_lr_scheduler:  # update by epoch 
                if self.lr_warmup_scheduler is None or self.lr_warmup_scheduler.finished():
                    self.lr_scheduler.step()

            self.last_epoch = epoch 

            if self.output_dir and epoch < self.train_dataloader.collate_fn.stop_epoch:
                checkpoint_paths = [self.output_dir / 'last.pth']
                # extra checkpoint before LR drop and every 100 epochs
                if (epoch + 1) % args.checkpoint_freq == 0:
                    checkpoint_paths.append(self.output_dir / f'checkpoint{epoch:04}.pth')
                for checkpoint_path in checkpoint_paths:
                    dist_utils.save_on_master(self.state_dict(), checkpoint_path)

            module = self.ema.module if self.ema else self.model
            test_stats, coco_evaluator = evaluate(
                module,
                self.criterion,
                self.postprocessor,
                self.val_dataloader,
                self.evaluator,
                self.device
            )

            need_load_resume_state = False
            for k in test_stats:
                if k not in best_stats:
                    best_stats[k] = {'epoch': -1, }
                if k not in best_stat_prints:
                    best_stat_prints[k] = {'epoch': -1, }
                if k not in top1s:
                    top1s[k] = 0
                best_stat = best_stats[k]
                best_stat_print = best_stat_prints[k]
                top1 = top1s[k]

                if self.writer and dist_utils.is_main_process():
                    for i, v in enumerate(test_stats[k]):
                        self.writer.add_scalar(f'Test/{k}_{i}'.format(k), v, epoch)

                if 'AP' in best_stat:
                    best_stat['epoch'] = epoch if test_stats[k][0] > best_stat['AP'] else best_stat['epoch']
                    best_stat['AP'] = max(best_stat['AP'], test_stats[k][0])
                else:
                    best_stat['epoch'] = epoch
                    best_stat['AP'] = test_stats[k][0]

                if best_stat['AP'] > top1:
                    best_stat_print['epoch'] = epoch
                    top1s[k] = best_stat['AP']
                    if self.output_dir:
                        if epoch >= self.train_dataloader.collate_fn.stop_epoch:
                            dist_utils.save_on_master(self.state_dict(), self.output_dir / (k + '_best_stg2.pth'))
                        else:
                            dist_utils.save_on_master(self.state_dict(), self.output_dir / (k + '_best_stg1.pth'))

                best_stat_print['AP'] = max(best_stat['AP'], top1)

                if best_stat['epoch'] == epoch and self.output_dir:
                    if epoch >= self.train_dataloader.collate_fn.stop_epoch:
                        if test_stats[k][0] > top1:
                            top1s[k] = test_stats[k][0]
                            dist_utils.save_on_master(self.state_dict(), self.output_dir / (k + '_best_stg2.pth'))
                    else:
                        top1s[k] = max(test_stats[k][0], top1)
                        dist_utils.save_on_master(self.state_dict(), self.output_dir / (k + '_best_stg1.pth'))

                elif epoch >= self.train_dataloader.collate_fn.stop_epoch and k == self.ema.testkey:
                    best_stats[k] = {'epoch': -1, }
                    need_load_resume_state = True
                    
            if need_load_resume_state:
                self.ema.decay -= 0.0001
                self.load_resume_state(str(self.output_dir / (self.ema.testkey + '_best_stg1.pth')))
                print(f'Refresh EMA at epoch {epoch} with decay {self.ema.decay}') 

            print(f'best_stat: {best_stat_prints}')  # global best


            log_stats = {
                **{f'train_{k}': v for k, v in train_stats.items()},
                **{f'test_{k}': v for k, v in test_stats.items()},
                'epoch': epoch,
                'n_parameters': n_parameters
            }

            if self.output_dir and dist_utils.is_main_process():
                with (self.output_dir / "log.txt").open("a") as f:
                    f.write(json.dumps(log_stats) + "\n")

                # for evaluation logs
                if coco_evaluator is not None:
                    if "bbox" in coco_evaluator.coco_eval:
                        (self.output_dir / 'eval' / 'bbox').mkdir(parents=True, exist_ok=True)
                        filenames = ['latest.pth']
                        if epoch % 50 == 0:
                            filenames.append(f'{epoch:03}.pth')
                        for name in filenames:
                            torch.save(coco_evaluator.coco_eval["bbox"].eval,
                                    self.output_dir / "eval" / "bbox" / name)
                    if 'segm' in coco_evaluator.coco_eval:
                        (self.output_dir / 'eval' / 'segm').mkdir(parents=True, exist_ok=True)
                        filenames = ['latest.pth']
                        if epoch % 50 == 0:
                            filenames.append(f'{epoch:03}.pth')
                        for name in filenames:
                            torch.save(coco_evaluator.coco_eval["segm"].eval,
                                    self.output_dir / "eval" / "segm" / name)

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('Training time {}'.format(total_time_str))


    def val(self, ):
        self.eval()

        module = self.ema.module if self.ema else self.model
        test_stats, coco_evaluator = evaluate(module, self.criterion, self.postprocessor,
                self.val_dataloader, self.evaluator, self.device)

        if self.output_dir:
            for k in coco_evaluator.coco_eval.keys():
                dist_utils.save_on_master(coco_evaluator.coco_eval[k].eval, self.output_dir / f"eval_{k}.pth")

        return
