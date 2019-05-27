# Environment

Overview of the environment and installation steps taken.

**NOTE:** the system used was running Windows 10 with Desbian WSL.

## Execution environment setup

The majority of the dependancies can be installed by simply installing `Python 3.7`, 
initializing a viritual environment, then running `pip install -r requirements.txt`.

### FastText

`FastText` was one of the packages that requires the most steps to install.
The following steps were taken to install it:

1. Clone the [repo](https://github.com/facebookresearch/fastText)
2. Build and install the project (as specified [here](https://github.com/facebookresearch/fastText/tree/master/python))
    1. **NOTE:** This required the installation of [VS C++ build tools](https://visualstudio.microsoft.com/downloads/#build-tools-for-visual-studio-2017)
    2. When building ensure you are in your projects virtual environment, if you're using PyCharm you can simply use the 
    *Terminal* tab at the bottom to navigate into the fastText directory in order to build it.
3. Download the English `.bin` pre-trained [model](https://fasttext.cc/docs/en/crawl-vectors.html)
    1. **NOTE:** The simple `.txt` model can be used if you don't have any out-of-vocabulary (OOV) words
