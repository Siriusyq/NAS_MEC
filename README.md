# How to use / evaluate 
* Use
    ```python
    # pytorch 
    from proxyless_nas import proxyless_cpu, proxyless_gpu, proxyless_mobile, proxyless_mobile_14, proxyless_cifar
    net = proxyless_cpu(pretrained=True) # Yes, we provide pre-trained models!
    ```
    ```python
    # tensorflow
    from proxyless_nas_tensorflow import proxyless_cpu, proxyless_gpu, proxyless_mobile, proxyless_mobile_14
    tf_net = proxyless_cpu(pretrained=True)
    ```

    If the above scripts failed to download, you download it manually from [Google Drive](https://drive.google.com/drive/folders/1qIaDsT95dKgrgaJk-KOMu6v9NLROv2tz?usp=sharing) and put them under  `$HOME/.torch/proxyless_nas/`.

* Evaluate

    `python eval.py --path 'Your path to imagent' --arch proxyless_cpu  # pytorch ImageNet`
    
    `python eval.py -d cifar10 # pytorch cifar10 `
    
    `python eval_tf.py --path 'Your path to imagent' --arch proxyless_cpu  # tensorflow`


## File structure

* [search](./search): code for neural architecture search.
* [training](./training): code for training searched models.
* [proxyless_nas](./proxyless_nas): pretrained models for PyTorch.



