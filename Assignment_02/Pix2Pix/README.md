### Implement [Pix2Pix](https://phillipi.github.io/pix2pix/) with [Fully Convolutional Layers](https://arxiv.org/abs/1411.4038)

Fill in the **Fully Convolutional Network** part of `FCN_network.py`, then prepare the dataset and train the model.

On **Linux / macOS**, the dataset can be prepared and the training can be started with:

```bash
bash download_facades_dataset.sh
cd Pix2Pix
python train.py
```

On **Windows**, `bash download_facades_dataset.sh` may not work directly in PowerShell, especially when **WSL is not installed**. In this case, it is recommended to manually download and prepare the Facades dataset with PowerShell commands, and then run the training script normally.

The provided code trains the model on the [Facades Dataset](https://cmp.felk.cvut.cz/~tylecr1/facade/). To achieve better generalization on the validation set, larger datasets from the [official Pix2Pix dataset collection](https://github.com/phillipi/pix2pix#datasets) can also be used.

For **Windows users**, the dataset can be prepared with the following PowerShell commands:

```powershell
New-Item -ItemType Directory -Force -Path .\Pix2Pix\datasets | Out-Null
Invoke-WebRequest -Uri "http://efrosgans.eecs.berkeley.edu/pix2pix/datasets/facades.tar.gz" -OutFile ".\Pix2Pix\datasets\facades.tar.gz"
tar -xzf .\Pix2Pix\datasets\facades.tar.gz -C .\Pix2Pix\datasets\
Remove-Item .\Pix2Pix\datasets\facades.tar.gz
Get-ChildItem .\Pix2Pix\datasets\facades\train -Filter *.jpg -Recurse | Sort-Object FullName | ForEach-Object { $_.FullName } | Set-Content .\Pix2Pix\train_list.txt
Get-ChildItem .\Pix2Pix\datasets\facades\val -Filter *.jpg -Recurse | Sort-Object FullName | ForEach-Object { $_.FullName } | Set-Content .\Pix2Pix\val_list.txt
```

After the dataset is prepared, run:

```bash
cd Pix2Pix
python train.py
```

After that, `train_list.txt` and `val_list.txt` will be generated automatically, and the training script will be able to read the dataset correctly.
