# ---> GENERAL CONFIG <---
name = 'panformer_GF-2'
description = 'test panformer on GF-2 dataset'
model_type = 'PanFormer'


# ---> DATASET CONFIG <---
max_iter = 200000
norm_input = False

# ---> SPECIFIC CONFIG <---
optim_cfg = {
    'core_module': dict(type='Adam', betas=(0.9, 0.999), lr=1e-4),
}
sched_cfg = dict(step_size=10000, gamma=0.99)
loss_cfg = {
    'rec_loss': dict(type='l1', w=1.)
}
model_cfg = {
    'core_module': dict(n_feats=64, n_heads=8, head_dim=8, win_size=4, n_blocks=3,
                        cross_module=['pan', 'ms'], cat_feat=['pan', 'ms']),
}