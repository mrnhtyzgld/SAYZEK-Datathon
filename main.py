from ultralytics import YOLO


## this code trained the latest 3 models
## i runned this overnight, latest model didnt work prob because ram

if __name__ == '__main__':

# eğittiğimizin üstüen etkrar dene
# lrf değiştirerek yap
# oto augment için üç ttane falan seçenek + augment parametresi


    model = YOLO("bestModels/yolov8m.pt")  # Modeli seçin
    model.train(data="sayzek.yaml", epochs=50, imgsz=512, device="0", batch=8,
                lrf=0.005,
                fliplr=0.5,
                #auto_augment='trivialaugment',
                augment=True,
                mosaic=0,
                mixup=0,
                flipud=0,
                copy_paste=0)
    results = model.val()

    model = YOLO("bestModels/yolov8x.pt")  # Modeli seçin
    model.train(data="sayzek.yaml", epochs=50, imgsz=512, device="0", batch=8,
                lrf=0.005,
                fliplr=0.5,
                # auto_augment='trivialaugment',
                augment=True,
                mosaic=0,
                mixup=0,
                flipud=0,
                copy_paste=0)
    results = model.val()

    model = YOLO("bestModels/yolov8m.pt")  # Modeli seçin
    model.train(data="sayzek.yaml", epochs=50, imgsz=512, device="0", batch=16,
                lrf=0.005,
                fliplr=0.5,
                # auto_augment='trivialaugment',
                augment=True,
                mosaic=0,
                mixup=0,
                flipud=0,
                copy_paste=0)
    results = model.val()

    model = YOLO("bestModels/yolov8x.pt")  # Modeli seçin
    model.train(data="sayzek.yaml", epochs=50, imgsz=512, device="0", batch=16,
                lrf=0.005,
                fliplr=0.5,
                # auto_augment='trivialaugment',
                augment=True,
                mosaic=0,
                mixup=0,
                flipud=0,
                copy_paste=0)
    results = model.val()


