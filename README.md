# MIR with TTA under Domain shift

## Dataset
Training
- IXI

Validation
- LPBA40
- CUMC12
- MGH10
- IBSR18

## Task
- Atlas-based registration
moving imageをatlasとし、fixed imageだけ可変の条件
少なくても輝度値ベースの割当がTest-Time Adaptationで対処できるか

- Inter-patient registration
解剖学的割当ができるか？
そもそもTest-Time Adaptationの有効性は、ある程度、学習時の知識をTTAでいじったら、テストにも使える前提だから、deep learning-based methodでの知識はsegmentationのラベルの一致を前提としているかも？？？

## Architecture
1. Classical registration method
- SyN

2. Deep Learning-based method
- TransMorph
