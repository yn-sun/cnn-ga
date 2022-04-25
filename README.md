# CNN-GA

Official implementation of the paper *Automatically Designing CNN Architectures Using the Genetic Algorithm for Image Classification*.

This code has been uodated for the adaption to torchvision 0.12 and pytorch 1.11
## Dependencies

- numpy
- torchvision 0.12
- pytorch 1.11

## Installation

1. Clone this repo by running `git clone https://github.com/yn-sun/cnn-ga`.
2. Alter hyperparameters in *global.ini*.
3. Run `python evolve.py`.

## Citing CNN-GA

It would be greatly appreciated if the following paper can be cited when the code has helped your research.

```
@article{sun2020automatically,
  title={Automatically designing CNN architectures using the genetic algorithm for image classification},
  author={Sun, Yanan and Xue, Bing and Zhang, Mengjie and Yen, Gary G and Lv, Jiancheng},
  journal={IEEE transactions on cybernetics},
  volume={50},
  number={9},
  pages={3840--3854},
  year={2020},
  publisher={IEEE}
}
```

