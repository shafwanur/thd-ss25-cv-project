1. **Download images**
   ```bash
   python download.py
   ```
   Downloads the dataset (1000 chicken images).

2. **Preprocess labels**
   ```bash
   python preprocess.py
   ```
   Converts raw labels to the required intermediate format.

3. **Convert to YOLO format**
   ```bash
   python csv_to_yolo_txt.py
   ```
   Converts preprocessed labels into YOLO-compatible `.txt` files.

4. **Train the model**
   ```bash
   python train_model.py
   ```
   Trains YOLO on the dataset (tested for 60 epochs).

5. **Test pretrained model**
   ```bash
   python single_test.py
   ```
   Runs inference using Denisa’s pretrained model (1000 images, 60 epochs).

## TL;DR

Skip training and just test the pretrained model:
```bash
python single_test.py
```
