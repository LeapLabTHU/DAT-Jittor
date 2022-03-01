# Vision Transformer with Deformable Attention (Jittor)

This repository contains a simple implementation of Vision Transformer with Deformable Attention \[[arXiv](https://arxiv.org/abs/2201.00520)\].

Currently, we only release the code of models and the training scripts are under development including advance data augmentations and mixed precision training.

Pytorch version: [Github](https://github.com/LeapLabTHU/DAT)


## Dependencies

- NVIDIA GPU + CUDA 11.1 + cuDNN 8.0.3
- Python 3.7 (Recommend to use Anaconda)
- jittor == 1.3.1.40
- [jimm](https://github.com/Jittor-Image-Models)


## TODO

- [ ] Training scripts with advance data augmentations.

## Citation

If you find our work is useful in your research, please consider citing:

```
@misc{xia2022vision,
      title={Vision Transformer with Deformable Attention}, 
      author={Zhuofan Xia and Xuran Pan and Shiji Song and Li Erran Li and Gao Huang},
      year={2022},
      eprint={2201.00520},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

### Contact

[xzf20@mails.tsinghua.edu.cn](xzf20@mails.tsinghua.edu.cn)