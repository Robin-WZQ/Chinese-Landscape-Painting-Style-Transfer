# Chinese-Landscape-Painting-style-transfer
Term project. Chinese Landscape Painting style transfer by adversarial network

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

2. Download pretrained model from BaiduYun . well, I trained the model for 100 epoches.

3. Pre-process the dataset

    ```Shell
    python canny.py
    python picture2texture.py
    python process_all.py
    ```

4. Configure your dataset and pretrained model path in
[opt.py](https://github.com/Robin-WZQ/Chinese-Landscape-Painting-Generation/opt.py).


    To train the model, please do:
    ```Shell
    python pix2pix.py
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
Also, I grap some pictures(nearly 800) from website, I added these pictures into the Alice dataset.
U can ran grap.py to generate you own dataset.
