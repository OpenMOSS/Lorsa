# Lorsa (Low-Rank Sparse Attention)

**Lorsa** is a novel attention decomposition method designed to tackle attention superposition, extracting tens of thousands of interpretable attention units from the attention layers of large language models.

## 📢 Important Notice

The complete implementation of Lorsa has been migrated to the [**Language-Model-SAEs**](https://github.com/OpenMOSS/Language-Model-SAEs) repository.

This repository is a comprehensive, fully-distributed framework for training, analyzing, and visualizing Sparse Autoencoders (SAEs) and their frontier variants, including:
- **Lorsa** (Low-Rank Sparse Attention)
- **CLT** (Cross-layer Transcoder)
- **MoLT** (Mixture of Linear Transforms)
- **CrossCoder**
- And many more SAE variants

## 🔗 Links

- **Code Implementation**: [OpenMOSS/Language-Model-SAEs](https://github.com/OpenMOSS/Language-Model-SAEs)
- **Paper**: [Towards Understanding the Nature of Attention with Low-Rank Sparse Decomposition](https://arxiv.org/abs/2504.20938)

## 📖 About Lorsa

Lorsa employs low-rank sparse decomposition to decompose attention layer outputs into interpretable feature units, effectively addressing the feature superposition problem in attention mechanisms. This enables a deeper understanding of how attention mechanisms work in large language models.

### Key Features

- Extract tens of thousands of interpretable attention units from attention layers
- Leverage low-rank sparse decomposition to handle attention superposition
- Support large-scale distributed training
- Provide comprehensive visualization tools

## 🚀 Quick Start

Please visit the [Language-Model-SAEs](https://github.com/OpenMOSS/Language-Model-SAEs) repository for:
- Installation instructions
- Training examples
- Usage tutorials

## 📚 Citation

If you use Lorsa in your research, please cite:

```bibtex
@article{He2025Lorsa,
  author    = {Zhengfu He and Junxuan Wang and Rui Lin and Xuyang Ge and 
               Wentao Shu and Qiong Tang and Junping Zhang and Xipeng Qiu},
  title     = {Towards Understanding the Nature of Attention with Low-Rank Sparse Decomposition},
  journal   = {CoRR},
  volume    = {abs/2504.20938},
  year      = {2025},
  url       = {https://arxiv.org/abs/2504.20938},
  eprint    = {2504.20938},
  eprinttype = {arXiv}
}
```

## 📝 Changelog

- **2025.4.29**: Initial release of Lorsa, introducing low-rank sparse attention decomposition
- **2025.11.9**: Implementation migrated to Language-Model-SAEs repository for better framework support

---

For any questions or suggestions, please submit an Issue in the [Language-Model-SAEs](https://github.com/OpenMOSS/Language-Model-SAEs) repository!
