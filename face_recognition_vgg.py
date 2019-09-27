import cv2
import numpy as np
from time import sleep
import keras
from keras.models import Model, Input
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
from keras.preprocessing import image

#cascade_path = "./models/haarcascade_frontalface_default.xml"
cascade_path = "./models/haarcascade_frontalface_alt2.xml"

def cv_fourcc(c1, c2, c3, c4):
        return (ord(c1) & 255) + ((ord(c2) & 255) << 8) + \
            ((ord(c3) & 255) << 16) + ((ord(c4) & 255) << 24)

def yomikomi(model,img):
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    preds = model.predict(preprocess_input(x))
    #preds = model.predict(x)
    results = decode_predictions(preds, top=1)[0]
    return str(results[0][1])

def model_definition():
    batch_size = 2
    num_classes = 1000
    img_rows, img_cols=224,224
    input_tensor = Input((img_rows, img_cols, 3))
    # 学習済みのVGG16をロード
    # 構造とともに学習済みの重みも読み込まれる
    model = VGG16(weights='imagenet', include_top=True, input_tensor=input_tensor)
    model.summary()
    return model

def main():
    #カスケード分類器の特徴量を取得する
    cascade = cv2.CascadeClassifier(cascade_path)
    color = (255, 255, 255) #白

    OUT_FILE_NAME = "./outputs/face_recognition.avi"
    FRAME_RATE=1
    w=224 #1280
    h=224 #960
    out = cv2.VideoWriter(OUT_FILE_NAME, \
          cv_fourcc('M', 'P', '4', 'V'), \
          FRAME_RATE, \
         (w, h), \
         True)    

    cap = cv2.VideoCapture(1)

    is_video = 'False'
    s=0.1
    model=model_definition()
    while True:
        timer = cv2.getTickCount()
        ret, frame = cap.read()
        sleep(s)
        fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer);
        # Display FPS on frame
        cv2.putText(frame, "FPS : " + str(int(1000*fps)), (100,50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50), 2);
        
        #グレースケール変換
        image_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #image_gray = cv2.equalizeHist(image_gray)
        facerect = cascade.detectMultiScale(image_gray, scaleFactor=1.1, minNeighbors=2, minSize=(30, 30))
        if len(facerect) > 0:
            #検出した顔を囲む矩形の作成
            for rect in facerect:
                y,x=tuple(rect[0:2])
                h,w=tuple(rect[2:4])
                cv2.rectangle(frame, tuple(rect[0:2]),tuple(rect[0:2]+rect[2:4]), color, thickness=2)
                roi = frame[y:y+h, x:x+w]
                try:
                    roi = cv2.resize(roi, (int(224), 224))
                    txt=yomikomi(model,roi)
                except:
                    txt=""
                    continue
                cv2.putText(frame, txt, (y,x), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (150,170,150), 2);
        cv2.imshow('test',frame)
        key = cv2.waitKey(1)&0xff
        
        if is_video=="True":
            img_dst = cv2.resize(frame, (int(224), 224)) #1280x960
            out.write(img_dst)
            print(is_video)

        if key == ord('q'):   #113
            #cv2.destroyAllWindows()
            break
        elif key == ord('p'):
            s=0.5
            is_video = "True"
        elif key == ord('s'):
            s=0.1
            is_video = "False"    
        
if __name__ == '__main__':
    main()
        
