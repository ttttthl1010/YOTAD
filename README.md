<!-- <img width="200px" src="https://github.com/shenscore/HTAD/blob/master/doc/logo.png" /> -->

# Introduction
YOTAD is a supervised  framework for chromatin domain (TAD) detection from Hi-C contact matrices.

# Citation

# Installing YOTAD
We recommend using [conda](https://github.com/conda/conda) to install YOTAD.
```
$ git clone https://github.com/ttttthl1010/YO-TAD
$ cd YO-TAD
$ conda create -n YOTAD python=3.11
$ conda activate YOTAD
$ pip install -r requirements.txt
```
# Running YOTAD
### (i) Data Processing
Convert the original cool files and bed files into YOLO format labels and images.
```
python ../ultralytics/detect/data.py
```

### (ii) Training
Train using the tags and images generated earlier.
```
python ../ultralytics/detect/train.py
```
### (iii) TAD Predicting
given the well trained TAD model file model.pt, run:
```
python pre.py 
```

**Output:**
+ The final TAD identification result in [BED](https://genome.ucsc.edu/FAQ/FAQformat.html#format1) format.

For visualization, we recommend utilizing [pyGenomeTracks](https://github.com/deeptools/pyGenomeTracks) or [juicebox](https://github.com/aidenlab/Juicebox).



# Contact us

**Hailin Tao**: taohailin1010@163.com <br>
