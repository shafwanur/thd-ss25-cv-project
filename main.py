import fiftyone
import fiftyone.zoo as foz

dataset = fiftyone.zoo.load_zoo_dataset(
              "open-images-v7",
              split="train",
              label_types=["detections"],
              classes=["Chicken"],
              max_samples=1000,
          )

# session = fiftyone.launch_app(dataset) # you don't necessarily need to launch the app. 
                                         # setting the database should start downloading the images already. 
                                         # if it fails, try running this the code with this line uncommented in jupyter notebook.
