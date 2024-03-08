<!-- Demo images are provided here: `dataset/data/train/GSO_demo`.

During TTT, we use Objaverse objects that is share similar CLIP feature with the in-the-wild object as 'super-class' regularization objects. You need to get them from google drive(https://drive.google.com/file/d/1usCRP4Cw0VRGnrOJ3ggHX4vRnWdLcld1/view?usp=sharing), decompress them and place them here : `dataset/data/objaverse`  -->
### Demo Images
We have made demo images readily accessible within the following directories: `dataset/data/train/GSO_demo` and `dataset/data/train/ABO_demo`. Our framework offers direct support for the GSO and ABO datasets. And if you're looking to play around with different datasets, no worries! You can either arrange them to match our setup or simply use our handy Gradio interface at `demo/sap3d` to load and showcase your images with ease.

### Using Objaverse Objects with TTT
During the Test Time Training(TTT) process, we utilize Objaverse objects. These objects share similar CLIP features with 'in-the-wild' objects and serve as 'super-class' regularization objects.

#### Obtaining Objaverse Objects
1. Download the Objaverse objects from Google Drive: [Download Objaverse Objects](https://drive.google.com/file/d/1usCRP4Cw0VRGnrOJ3ggHX4vRnWdLcld1/view?usp=sharing).
2. Decompress the downloaded file.
3. Place the decompressed files in the following directory: `dataset/data/objaverse`.
