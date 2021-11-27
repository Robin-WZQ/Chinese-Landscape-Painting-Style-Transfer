# Chinese-Landscape-Painting-style-transfer
Term project. Chinese Landscape Painting style transfer by adversarial network

还在修改中，进度80% (this is a unstable version, I will finish it before 2021/12/17 )

## Result
![图片1](https://github.com/Robin-WZQ/Chinese-Landscape-Painting-Generation/blob/main/assets/result.png)


## Installation
The code was tested with Anaconda and Python 3.7. After installing the Anaconda environment:

0. Clone the repo:
    ```Shell
    git clone https://github.com/Robin-WZQ/Chinese-Landscape-Painting-Generation.git
    cd Chinese-Landscape-Painting-Generation
    ```

1. Install dependencies:

    For PyTorch dependency, see [pytorch.org](https://pytorch.org/) for more details.

    For custom dependencies:
    ```Shell
    pip install -r requirements.txt
    ```

2. Download pretrained model from [BaiduYun](https://pan.baidu.com/s/1_wSIMuMBNj4g2BKE0a_Okg) extract code: qls1
well, I trained the model for 100 epoches.

3. Pre-process the dataset

    ```Shell
    python canny.py
    python picture2texture.py
    python process_all.py
    ```

4. Configure your dataset and pretrained model path in
[opts.py](https://github.com/Robin-WZQ/Chinese-Landscape-Painting-Generation/opts.py).


    To train the model, please do:
    ```Shell
    python pix2pix.py
    ```
    
    To do a style trasfer, please do:
    ```Shell
    python test.py
    ```
    
## Datasets:

I used the dataset from [here](https://github.com/alicex2020/Chinese-Landscape-Painting-Dataset)

Dataset directory tree is shown below

- **Alice**
Make sure to put the files as the following structure:
  ```
  Alice
  ├── Harvard
  │   ├── Harvard_0.jpg
  │   └── ...
  ├── met-1
  │   ├── met_0.jpg
  │   └── ...
  └── met-2
  │   ├── met-221.jpg
  │   └── ...
  ```
Also, I graped some pictures(nearly 800) from website and I added these pictures into the Alice dataset.
U can ran tools/Web_Spider.py to generate you own dataset.

After preprocessing, it will generate 3 dataset:
- processed by canny, like:

<img src="https://github.com/Robin-WZQ/Chinese-Landscape-Painting-Generation/blob/main/assets/harvard_65_0.jpg" width="256px">
- processed by HED, like:

<img src="https://github.com/Robin-WZQ/Chinese-Landscape-Painting-Generation/blob/main/assets/harvard_65_1.jpg" width="256px">
- canny + HED, like:

<img src="https://github.com/Robin-WZQ/Chinese-Landscape-Painting-Generation/blob/main/assets/harvard_65.jpg" width="256px">

## Experiments
These models were trained in machine with NVIDIA TITAN X 11gb GPU. I trained it in 100 epoches and bachsize is 1. More details please see in my code.

## Code Tree
