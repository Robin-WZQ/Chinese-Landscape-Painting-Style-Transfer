# Chinese-Landscape-Painting-style-transfer
Chinese Landscape Painting style transfer by adversarial network.


![图片2](https://github.com/Robin-WZQ/Chinese-Landscape-Painting-Generation/blob/main/assets/test.gif)

## Result
![图片1](https://github.com/Robin-WZQ/Chinese-Landscape-Painting-Generation/blob/main/assets/results.jpg)


## Installation
The code was tested with Anaconda and Python 3.7. After installing the Anaconda environment:

0. Clone the repo:
    ```Shell
    git clone https://github.com/Robin-WZQ/Chinese-Landscape-Painting-Style-Transfer.git
    cd Chinese-Landscape-Painting-Style-Transfer
    ```

1. Install dependencies:

    For PyTorch dependency, see [pytorch.org](https://pytorch.org/) for more details.

    For custom dependencies:
    ```Shell
    pip install -r requirements.txt
    ```

2. Download pretrained model from [BaiduYun](https://pan.baidu.com/s/1_wSIMuMBNj4g2BKE0a_Okg) extract code: qls1

    I trained the model for 100 epoches.

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
    
5. In the folder UI, I make a simple interactive interface by tkinter. 

    *This is a separate folder. Do not use it in the same directory as other folders！！* 

    To run the DEMO, please do:
    ```Shell
    python UI/UI.py
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
U can run tools/Web_Spider.py to generate you own dataset. (And modify names in tools/name.txt)

After preprocessing, it will generate 3 dataset:
- processed by canny (I use Histogram Equalization for image enhancement first), like:

<img src="https://github.com/Robin-WZQ/Chinese-Landscape-Painting-Generation/blob/main/assets/harvard_65_0.jpg" width="256px">
- processed by HED, like:

<img src="https://github.com/Robin-WZQ/Chinese-Landscape-Painting-Generation/blob/main/assets/harvard_65_1.jpg" width="256px">
- canny + HED, like:

<img src="https://github.com/Robin-WZQ/Chinese-Landscape-Painting-Generation/blob/main/assets/harvard_65.jpg" width="256px">

## Experiments
These models were trained in machine with NVIDIA TITAN X 11gb GPU. I trained it in 100 epoches and bachsize is 1. More details please see in my code.

## Future Work
In the future, I will continue to study related topics. In my recent chat with painters, I got new inspiration about the style-transfer and generation of traditional Chinese painting, involving many tasks such as learning semantic segmentation, dataset establishment and so on. I hope I can realize it within a year.

---------------------------update in 2022/4/29 -------------------------------

I made it! This is my new work: https://github.com/Robin-WZQ/Xi-Meng
