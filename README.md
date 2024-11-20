# Sarawak Deforestation Drivers Classification

This repository contains the implementation of deep learning models for classifying drivers of deforestation in Sarawak, Malaysia, using multi-temporal Landsat imagery and topographic data. The research explores various deep learning architectures including ResNet, EfficientNet, UNet, and transformer-based models to classify deforestation events into four main categories.

## Research Overview

The study investigates the application of transfer learning by training models on Indonesian deforestation data and applying them to classify deforestation drivers in Sarawak. Our EfficientNet implementation achieved 65% accuracy on the test dataset, successfully identifying plantations as the predominant driver of deforestation in the region.

### Key Features

- Multi-temporal Landsat 8 imagery analysis
- Integration of SRTM topographic data
- Implementation of multiple deep learning architectures
- Transfer learning from Indonesian to Malaysian context
- Temporal analysis of deforestation patterns (2017-2018)

## Models Implemented

- ResNet-18
- EfficientNet-B2
- UNet
- Vision Transformer (ViT)
- TransResNet (Hybrid CNN-Transformer)

## Dataset

The study utilizes two main datasets:

1. Training Dataset (Indonesia):
   - 632 expertly annotated forest loss events
   - Four classification categories: Plantation, Smallholder agriculture, Grassland/Shrubland, and Other
   - Source: Austin et al. (2019)

2. Test Dataset (Sarawak):
   - 140 forest loss events (2017-2018)
   - Includes complete four-year Landsat 8 time series
   - Incorporates slope information from SRTM data

## Requirements

```
python>=3.8
torch>=1.8.0
torchvision>=0.9.0
rasterio>=1.2.0
numpy>=1.19.0
pandas>=1.2.0
scikit-learn>=0.24.0
```

## Installation

```bash
git clone https://github.com/AbdullahiHD/Sarawak-Deforestation-Drivers-Classification.git
cd Sarawak-Deforestation-Drivers-Classification
pip install -r requirements.txt
```

## Usage
1. Data Collection
``` JavaScript
  run GFC_DATA.js and Satellite_Imagery.js
```
1. Data Preparation:
```python
 run Training + Effnet + Resnet.ipynb or Training - UNet.ipynb or Training ViT Transformer.ipynb
```

2. Model Training:
```python
 run Training + Effnet + Resnet.ipynb or Training - UNet.ipynb or Training ViT Transformer.ipynb
```

3. Inference:
```python
 run PredictInsights.ipynb
```

## Results

Model performance on Indonesia test dataset:

| Model | Accuracy |
|-------|----------|
| EfficientNet | 0.65 |
| UNet | 0.51 |
| ResNet | 0.49 |
| Vision Transformer | 0.48 |
| TransResNet | 0.46 |

## Limitations

1. Lack of region-specific training data for Sarawak
2. Limited ground truth validation data
3. Annual temporal resolution may miss finer-scale changes
4. Simplified classification scheme (4 categories)
5. Limited integration of socio-economic variables

## Future Work

- Exploration of advanced transformer architectures
- Integration of higher resolution satellite imagery
- Incorporation of additional data sources (radar, multispectral)
- Development of more comprehensive validation datasets
- Extension to other regions in Southeast Asia

## Citation

If you use this code in your research, please cite our paper:

```bibtex
@article{dahir2024remote,
  title={Remote-sensing Data-driven Classification of Deforestation based on Machine Learning in Sarawak},
  author={Dahir, Abdullahi Hussein and Chang, Miko MayLee},
  journal={Journal of Engineering Science and Technology},
  year={2024}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

For questions or collaboration opportunities, please contact:
- Abdullahi Hussein Dahir - 102778118@students.swinburne.edu.my
- Miko MayLee Chang - mchang@swinburne.edu.my

## Acknowledgments

- Swinburne University of Technology Sarawak Campus
- Contributors to the Global Forest Change dataset
- Authors of the Indonesian deforestation drivers dataset (Austin et al. (2019) & Ramachandran, N., et al. (2024))
