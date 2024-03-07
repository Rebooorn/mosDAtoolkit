
---

<div align="center">    
 
# Multi Organ Segmentation Data Augmentation Toolkit

</div>
 
## Description   
This toolkit is part of my PhD research about data augmentation (DA) strategies on multi-organ segmentaiton(MOS) (mainly CT). These strategies have been re-implemented for MOS:
- CutMix
- ObjectAug
- CarveMix
- AnatoMix

## Dependency
Some utils are from my another repo `mostoolkit`.  
In cases when inpaint utils are used, then you need to install `pytorch`.

## Some notes 
Use `data_preparation.ipynb` to preprocess the data for DA strategies and data split configs(maybe needed for DA). `prepare_nnunet_dataset.py` might be useful to run nnUNet. `metrics.py` is used to generate the evaluation metrics (micro/macro avaraged dice score)

## Examples
1. CutMix
```python
python cli_cutmix.py -sp split.json -d ./amos128 -s ./amoscutmix -n 200
```
2. ObjectAug
```python
python cli_objectaug.py -sp split.json -d ./amos128 -s ./amosobjectaug -n 200 -nc 16 -nw 8
```
3. CarveMix
```python
python cli_carvemix.py -sp split.json -d ./amos128 -s ./amoscarvemix -n 200 -nc 16 -nw 8
```
4. AnatoMix
```python 
python cli_anatomix_v2.py -sp split.json -d ./amos128 -s ./amosanatomix -n 200 -nc 16 -nw 8
```
