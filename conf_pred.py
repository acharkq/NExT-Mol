import os
import argparse
import torch
import warnings
import lightning as L
import lightning.pytorch.callbacks as plc
from lightning.pytorch.loggers import CSVLogger
from model.dist_funs import MyDeepSpeedStrategy
from model.train_unimol_conf import UniMolConfTrain
from data_provider.qm9_dm import QM9DM

os.environ['OPENBLAS_NUM_THREADS'] = '1'
## for pyg bug
warnings.filterwarnings('ignore', category=UserWarning, message='TypedStorage is deprecated')
## for A100 gpus
torch.set_float32_matmul_precision('medium') # can be medium (bfloat16), high (tensorfloat32), highest (float32)



def main(args):
    L.seed_everything(args.seed)

    # tokenizer
    tokenizer = UniMolConfTrain.init_tokenizer(args)
    print('before inserting total tokens:', len(tokenizer))
    dm = QM9DM(args.root, args.num_workers, args.batch_size, tokenizer)
    print('after inserting total tokens:', len(tokenizer))
    
    # model
    current_epoch = 0
    if args.init_checkpoint:
        model = UniMolConfTrain.load_from_checkpoint(args.init_checkpoint, device=args.devices, strict=False, args=args, tokenizer=tokenizer, max_sf_tokens=dm.max_sf_tokens)
        print(f"loading model from {args.init_checkpoint}")
        ckpt = torch.load(args.init_checkpoint, map_location='cpu')
        current_epoch = ckpt['epoch']
    else:
        model = UniMolConfTrain(args, tokenizer=tokenizer, max_sf_tokens=dm.max_sf_tokens)
    
    print('total params:', sum(p.numel() for p in model.parameters()))
    
    callbacks = []
    callbacks.append(plc.ModelCheckpoint(dirpath="all_checkpoints/"+args.filename+"/",
                                         filename='{epoch:02d}',
                                         every_n_epochs=args.save_every_n_epochs,
                                         save_top_k=-1,
                                         save_on_train_epoch_end=True))

    if len(args.devices.split(',')) > 1:
        if args.strategy_name == 'deepspeed':
            strategy = MyDeepSpeedStrategy(stage=2)
        else:
            strategy = L.strategies.DDPStrategy(start_method='spawn', find_unused_parameters=False)
    else:
        strategy = 'auto'
        args.devices = eval(args.devices)
        print(args.devices)
    logger = CSVLogger(save_dir=f'./all_checkpoints/{args.filename}/')
    
    if args.delta_train and args.mode == 'train':
        trainer = L.Trainer(
            accelerator=args.accelerator,
            devices=args.devices,
            precision=args.precision,
            max_epochs=10,
            accumulate_grad_batches=args.accumulate_grad_batches,
            check_val_every_n_epoch=5,
            strategy=strategy,
            logger=logger,
        )
        args.mode = 'delta_train'
        if args.unimol_version == 'v2':
            model.set_trainble_params(['projector', 'gbf_bt', 'gbf_proj_bt'])
        elif args.unimol_version == 'v3':
            model.set_trainble_params(['projector', 'gbf', 'gbf_proj'])
        elif args.unimol_version == 'v4':
            model.set_trainble_params(['projector', 'gbf', 'gbf_proj', 'in_proj_bond', 'bond_embed0', 'bond_embed1'])
        else:
            raise NotImplementedError()
        trainer.fit(model, datamodule=dm)
        model.restore_trainble_params()
        args.mode = 'train'

    
    trainer = L.Trainer(
        accelerator=args.accelerator,
        devices=args.devices,
        precision=args.precision,
        max_epochs=args.max_epochs,
        accumulate_grad_batches=args.accumulate_grad_batches,
        check_val_every_n_epoch=args.check_val_every_n_epoch,
        callbacks=callbacks,
        strategy=strategy,
        logger=logger,
        # limit_val_batches=100,
    )
    if args.mode in ['train']:
        trainer.fit(model, datamodule=dm)
        trainer.validate(model, datamodule=dm)
    elif args.mode in {'eval', 'eval_gen', 'eval_conf'}:
        trainer.fit_loop.epoch_progress.current.completed = current_epoch
        trainer.validate(model, datamodule=dm)
    else:
        raise NotImplementedError()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--filename', type=str, default="test")
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--strategy_name', type=str, default='deepspeed')
    parser.add_argument('--accelerator', type=str, default='gpu')
    parser.add_argument('--devices', type=str, default='0,1,2,3,4,5')
    parser.add_argument('--precision', type=str, default='bf16-mixed')
    parser.add_argument('--max_epochs', type=int, default=100)
    parser.add_argument('--check_val_every_n_epoch', type=int, default=1)
    parser.add_argument('--save_every_n_epochs', type=int, default=10)
    parser.add_argument('--accumulate_grad_batches', type=int, default=1)
    parser.add_argument('--delta_train', action='store_true', default=False)
    parser = UniMolConfTrain.add_model_specific_args(parser)
    parser = QM9DM.add_model_specific_args(parser)
    args = parser.parse_args()

    print("=========================================")
    for k, v in sorted(vars(args).items()):
        print(k, '=', v)
    print("=========================================")
    main(args)

