# Instructions to use each model

## PoseNet with Beta Loss

    ./rgb_training -fn posenet_beta_lab024_50k -mn posenetgooglenet -train_set seq41 -test_set seq42 -n_epochs 300 -batch_size 45  -loss 'BetaLoss(100)' -c -im 'PoseNetGoogleNet(True,0.8)' -lr_step_size 60 -lr 1e-4 -lr_gamma 0.5 -wd 1e-2 -gpu 2

## PoseNet with Dynamic Loss

    ./rgb_training -fn posenet_dynamic_p0r20 -mn posenetgooglenet -train_set seq21_p0r20 -test_set seq22_p0r20 -n_epochs 300 -batch_size 45  -loss 'DynamicLoss(sx=0,sq=-1)' -c -im 'PoseNetGoogleNet(True,0.6)' -lr_step_size 150 -lr 1e-4 -lr_gamma 0.5 -wd 1e-3 -gpu 0

## PoseNet with Resnet

    ./rgb_training -fn posenet_resnet_lab024_50k -mn posenetresnet -train_set seq41 -test_set seq42 -n_epochs 400 -batch_size 45  -loss 'DynamicLoss(sx=0,sq=-1)' -c -im 'PoseNetResNet(True,0.6)' -lr_step_size 150 -lr 1e-4 -lr_gamma 0.5 -wd 1e-3 -gpu 3

## PoseLSTM

    ./rgb_training -fn poselstm_san_50k -mn poselstm -train_set seq53 -test_set seq54 -n_epochs 500 -batch_size 45  -loss 'BetaLoss(100)' -c -im 'PoseLSTM()' -lr_step_size 150 -lr 1e-4 -lr_gamma 0.5 -wd 1e-3 -gpu 3

## Hourglass

    ./rgb_traning -fn hourglass_lab024_50k -mn hourglass -train_set seq41 -test_set seq42 -n_epochs 500 -batch_size 45  -loss 'BetaLoss(10)' -c -im 'HourglassBatch(pretrained=True,dropout_rate=0.5,sum_mode=False)' -lr_step_size 50 -lr 1e-4 -lr_gamma 0.5 -wd 1e-5 -gpu 2 -cs 224