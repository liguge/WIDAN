# Interpretable Physics-informed Domain Adaptation Paradigm for Cross-machine Transfer Diagnosis

The pytorch implementation of the paper [Interpretable Physics-informed Domain Adaptation Paradigm for Cross-machine Transfer Diagnosis](https://doi.org/10.1016/j.knosys.2024.111499)



# Result!

The code is based on UDTL and the PU dataset, and the accuracy can be improved from 63.79% to 77.19%, an increase of 13.40%.

- Baseline(**Line 183 in resnet_18.py**)

  ![image-20240405212227670](G:\研究生资料\博士\投稿论文\第三篇\image-20240405212227670.png)

- Weight initialization

  ![image-20240405212209638](G:\研究生资料\博士\投稿论文\第三篇\image-20240405212209638.png)

## Brief introduction  
While transfer learning-based intelligent diagnosis has achieved significant breakthroughs, the performance of existing well-known methods still needs urgent improvement, given the increasingly significant distribution discrepancy between source and target domain data from different machines. To tackle this issue, rather than designing domain discrepancy statistical metrics or elaborate network architecture, we delve deep into the interaction and mutual promotion between signal processing and domain adaptation. Inspired by wavelet technology and weight initialization, an end-to-end, succinct, and high-performance Physics-informed wavelet domain adaptation network (WIDAN) has been subtly devised, which integrates interpretable wavelet knowledge into the dual-stream convolutional layer with independent weights to cope with extremely challenging cross-machine diagnostic tasks. Specifically, the first-layer weights of a CNN are updated with optimized and informative Laplace or Morlet weights. This approach alleviates troublesome parameter selection, where scaling and translation factors with specific physical interpretations are constrained by the convolution kernel parameters. Additionally, a smooth-assisted scaling factor is introduced to ensure consistency with neural network weights. Furthermore, a dual-stream bottleneck layer is designed to learn reasonable weights to pre-transform different domain data into a uniform common space. This can promote WIDAN to extract domain-invariant features. Holistic evaluations confirm that WIDAN outperforms state-of-the-art models across multiple tasks, indicating that a wide first-layer kernel with optimized wavelet weight initialization can enhance domain transferability, thus validly fostering cross-machine transfer diagnosis.

## Highlights

- A first-layer kernel with wavelet weights is designed to diminish domain discrepancy.
- Optimized wavelet weights are devised as the first-layer initialization.
- The dual-stream module is designed to promote the domain transferability.
- A paradigm for transfer diagnosis is posed to design the first convolutional layer.
- The availability of WIDAN is validated with four data sources.


## Paper
Interpretable Physics-informed Domain Adaptation Paradigm for Cross-machine Transfer Diagnosis

Chao He<sup>a,b</sup>, **Hongmei Shi<sup>a,b,*</sup>**, Xiaorong Liu<sup>c</sup> and Jianbo Li<sup>a,b</sup>

<sup>a</sup>State Key Laboratory of Advanced Rail Autonomous Operation, Beijing Jiaotong University, Beijing 100044, China 

<sup>b</sup>School of Mechanical, Electronic and Control Engineering, Beijing Jiaotong University, Beijing 100044, China

<sup>c</sup>School of Computing and Artificial Intelligence, Southwest Jiaotong University, Chengdu 611756, China

[Knowledge-Based Systems](https://www.sciencedirect.com/journal/knowledge-based-systems/vol/288/suppl/C)



## Citation

```html
@article{he2024interpretable,
  title={Interpretable physics-informed domain adaptation paradigm for cross-machine transfer diagnosis},
  author={He, Chao and Shi, Hongmei and Liu, Xiaorong and Li, Jianbo},
  journal={Knowledge-Based Systems},
  volume={288},
  pages={111499},
  year={2024},
  doi={10.1016/j.knosys.2024.111499}
}
```

C. He, H. Shi, X. Liu, J. Li, Interpretable Physics-informed Domain Adaptation Paradigm for Cross-machine Transfer Diagnosis, Knowledge-Based Systems 288 (2024) 111499, https://doi.org/10.1016/j.knosys.2024.111499




## Ackowledgements
The authors are grateful for the supports of the National Natural Science Foundation of China (No. 52272429), and State Key Laboratory of Advanced Rail Autonomous Operation (Contract No. RAO2023ZZ003).



## Contact

- **Chao He**
- **chaohe#bjtu.edu.cn (please replace # by @)**

​      
