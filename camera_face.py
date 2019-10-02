#https://qiita.com/daiarg/items/3ea91b08f0d1cb5bfc61

import cv2
import glob
import time
import sys
from datetime import  datetime

cap=cv2.VideoCapture(1) #0にするとnotePCのカメラ、1にすると外付けのUSBカメラにできる

# 顔判定で使うxmlファイルを指定する。(opencvのpathを指定)
cascade_path = "./models/haarcascade_frontalface_alt2.xml"
cascade = cv2.CascadeClassifier(cascade_path)
color = (255, 255, 255) #白
path = "./img/" # 写真を格納するフォルダを指定

num=300 # 欲しいファイルの数
label = str(input("人を判別する数字を入力してください ex.0："))
file_number = len(glob.glob(path+label+"/*")) #現在のフォルダ内のファイル数
count = 0 #撮影した写真枚数の初期値
sk=0
while True:
    #フォルダの中に保存された写真の枚数がnum以下の場合は撮影を続ける
    if count < num:
        time.sleep(0.01) #cap reflesh
        print("あと{}枚です".format(num-count))

        now = datetime.now()#撮影時間
        r, img = cap.read()

        # 結果を保存するための変数を用意しておく
        img_result = img

        # グレースケールに変換
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        #顔判定 minSize で顔判定する際の最小の四角の大きさを指定できる。(小さい値を指定し過ぎると顔っぽい小さなシミのような部分も判定されてしまう。)
        faces=cascade.detectMultiScale(img_gray, scaleFactor=1.1, minNeighbors=1, minSize=(30, 30))

        # 顔があった場合
        if len(faces) > 0:
            # 複数の顔があった場合、１つずつ四角で囲っていく
            for rect in faces:
                x=rect[0]
                y=rect[1]
                width=rect[2]
                height=rect[3]
                #print(y,x,y2,x2)
                roi = img[y:y+height, x:x+width]  #frame[y:y+h, x:x+w]
                cv2.rectangle(img, tuple(rect[0:2]),tuple(rect[0:2]+rect[2:4]), color, thickness=2)
                cv2.imshow("img",img)
                try:
                    roi = cv2.resize(roi, (int(224), 224))
                    cv2.imshow('roi',roi)
                    cv2.imwrite(path+"/"+label+"/"+str(sk)+'.jpg', roi)
                except:
                    txt=""
                    continue
                
                sk +=1

        #現在の写真枚数から初期値を減算して、今回撮影した写真の枚数をカウント
        count = len(glob.glob(path+"/"+label+"/*")) - file_number
        key = cv2.waitKey(1)&0xff
        if key == ord('q'):   #113
            #cv2.destroyAllWindows()
            break

    #フォルダの中に保存された写真の枚数がnumを満たしたので撮影を終える
    else:
        break

#カメラをOFFにする
cap.release()