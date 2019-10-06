# coding: utf-8
from pykakasi import kakasi

import cv2
import numpy as np
from time import sleep
import keras
from keras.models import Model, Input
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
from keras.preprocessing import image
from PIL import ImageFont, ImageDraw, Image
import pyaudio
import wave
import time
import re
import csv

#cascade_path = "./models/haarcascade_frontalface_default.xml"
cascade_path = "./models/haarcascade_frontalface_alt2.xml"

def cv_fourcc(c1, c2, c3, c4):
        return (ord(c1) & 255) + ((ord(c2) & 255) << 8) + \
            ((ord(c3) & 255) << 16) + ((ord(c4) & 255) << 24)

def text2speak(num0):
    RATE=44100
    CHUNK = 22050
    p=pyaudio.PyAudio()
    kakasi_ = kakasi()

    sentence=num0

    kakasi_.setMode('J', 'H')  # J(Kanji) to H(Hiragana)
    kakasi_.setMode('H', 'H') # H(Hiragana) to None(noconversion)
    kakasi_.setMode('K', 'H') # K(Katakana) to a(Hiragana)

    conv = kakasi_.getConverter()

    char_list = list(conv.do(sentence))

    kakasi_.setMode('H', 'a') # H(Hiragana) to a(roman)
    conv = kakasi_.getConverter()
    sentences=[]
    for i in range(len(char_list)):
        sent= conv.do(char_list[i])
        sentences.append(sent)
    
    f_list=[]
    f_list=sentences

    stream=p.open(format = pyaudio.paInt16,
        channels = 1,
        rate = int(RATE*1.8),
        frames_per_buffer = CHUNK,
        input = True,
        output = True) # inputとoutputを同時にTrueにする

    w = wave.Wave_write("./pyaudio/ohayo005_sin.wav")
    p = (1, 2, RATE, CHUNK, 'NONE', 'not compressed')
    w.setparams(p)
    for i in f_list:
        i = re.sub(r"[^a-z]", "", i)
        if i== '':
            continue
        else:
            wavfile = './pyaudio/aiueo/'+i+'.wav'
        #print(wavfile)
        try:
            wr = wave.open(wavfile, "rb")
        except:
            wavfile = './pyaudio/aiueo/n.wav'
            continue
        input = wr.readframes(wr.getnframes())
        output = stream.write(input)
        w.writeframes(input)
    
def yomikomi(model,img,num_classes=3):
    if num_classes==4:
        name_list=['えがお','へんがお','まじめ','悲しい','セクシー']
    else:
        name_list=['まゆゆ','あかりんだーすー','ウワン']
        
    x = img/255 #image.img_to_array(img)
    cv2.imshow("x",x)
    x = np.expand_dims(x, axis=0)
    preds = model.predict(x)
    index=np.argmax(preds[0])
    print(preds[0])
    print(index,name_list[index],preds[0][index])
    return name_list[index], preds[0][index]

def model_definition(num_classes = 4,params='params4_model_VGG16L3_i_100.hdf5'):
    batch_size = 2
    num_classes = num_classes
    img_rows, img_cols=224,224
    input_tensor = Input((img_rows, img_cols, 3))
    # 学習済みのVGG16をロード
    # 構造とともに学習済みの重みも読み込まれる
    vgg16 = VGG16(weights='imagenet', include_top=False, input_tensor=input_tensor)
    
    # FC層を構築
    top_model = Sequential()
    top_model.add(Flatten(input_shape=vgg16.output_shape[1:])) #vgg16,vgg19,InceptionV3, ResNet50
    top_model.add(Dense(256, activation='relu'))
    top_model.add(Dropout(0.5))
    top_model.add(Dense(num_classes, activation='softmax'))

    # VGG16とFCを接続
    model = Model(input=vgg16.input, output=top_model(vgg16.output))
    model.summary()
    model.load_weights(params)
    return model

def egao(txt_parson,txt_egao):
    if txt_egao=='えがお':
        num0=txt_parson+'さん'+txt_egao+"がかわいいね"
    elif txt_egao=='へんがお':
        num0=txt_parson+'さん'+txt_egao+"笑っちゃう"
    elif txt_egao=='まじめ':
        num0=txt_parson+'さん'+txt_egao+"な顔もいいね"
    elif txt_egao=='悲しい':
        num0=txt_parson+'さん'+"何か"+ txt_egao +"ことでもあったの" 
    elif txt_egao=='セクシー':
        num0=txt_parson+'さん'+txt +"だね" 
    print(num0)
    
    return num0
    

def main():
    kakasi_ = kakasi()
    kakasi_.setMode('H', 'a')
    kakasi_.setMode('K', 'a')
    kakasi_.setMode('J', 'a')
    conv = kakasi_.getConverter()
    
    #カスケード分類器の特徴量を取得する
    cascade = cv2.CascadeClassifier(cascade_path)
    color = (255, 255, 255) #白
    path = "./img/"
    label = str(input("人を判別する数字を入力してください ex.0："))

    OUT_FILE_NAME = "./img/face_recognition.avi"
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
    s=1
    model_egao=model_definition(num_classes = 4,params='params4_model_VGG16L3_i_100.hdf5')
    model_egao_m=model_definition(num_classes = 4,params='params4m_model_VGG16L3_i_100.hdf5')
    model_parson=model_definition(num_classes = 3,params='history_params_model_VGG16L3_i_100.hdf5')
    ## Use HGS創英角ゴシックポップ体標準 to write Japanese.
    fontpath ='C:\Windows\Fonts\HGRPP1.TTC' # Windows10 だと C:\Windows\Fonts\ 以下にフォントがあります。
    font = ImageFont.truetype(fontpath, 16) # フォントサイズが32
    font0 = cv2.FONT_HERSHEY_SIMPLEX
    sk=0
    while True:
        b,g,r,a = 0,255,0,0 #B(青)・G(緑)・R(赤)・A(透明度)
        timer = cv2.getTickCount()
        ret, frame = cap.read()
        sleep(s)
        fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer);
        # Display FPS on frame
        cv2.putText(frame, "FPS : " + str(int(1000*fps)), (100,50), font0, 0.75, (50,170,50), 2);
        #グレースケール変換
        image_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #image_gray = cv2.equalizeHist(image_gray)
        facerect = cascade.detectMultiScale(image_gray, scaleFactor=1.1, minNeighbors=2, minSize=(30, 30))
        #print(len(facerect))
        img=frame
        if len(facerect) > 0:
            #検出した顔を囲む矩形の作成
            for rect in facerect:
                x=rect[0]
                y=rect[1]
                width=rect[2]
                height=rect[3]
                roi = img[y:y+height, x:x+width]  #frame[y:y+h, x:x+w]
                cv2.rectangle(img, tuple(rect[0:2]),tuple(rect[0:2]+rect[2:4]), color, thickness=2)
                
                try:
                    roi = cv2.resize(roi, (int(224), 224))
                    cv2.imshow('roi',roi)
                    txt_parson, preds_parson=yomikomi(model_parson,roi,num_classes=3)
                    if txt_parson=="まゆゆ":
                        txt_egao, preds_egao=yomikomi(model_egao_m,roi,num_classes=4)
                    else:
                        txt_egao, preds_egao=yomikomi(model_egao,roi,num_classes=4)
                    print("txt_egao, preds",txt_egao,preds_egao*100 ," %")
                    print("txt_parson, preds",txt_parson,preds_parson*100 ," %")
                    txt2=conv.do(txt_parson+txt_egao)
                    cv2.imwrite(path+"/"+label+"/"+str(sk)+'_'+str(txt2)+'_'+str(int(preds_parson*100))+'_'+str(int(preds_egao*100))+'.jpg', roi)
                    
                    num0=egao(txt_parson,txt_egao)
                    text2speak(num0)
                    txt=txt_parson+txt_egao
                    #time.sleep(2)
                    img_pil = Image.fromarray(img) # 配列の各値を8bit(1byte)整数型(0～255)をPIL Imageに変換。
                    draw = ImageDraw.Draw(img_pil) # drawインスタンスを生成
                    position = (x, y) # テキスト表示位置
                    draw.text(position, txt, font = font , fill = (b, g, r, a) ) # drawにテキストを記載 fill:色 BGRA (RGB)
                    img = np.array(img_pil) # PIL を配列に変換
                except:
                    txt=""
                    continue
        cv2.imshow('test',img)
        sk +=1
        key = cv2.waitKey(1)&0xff
        
        if is_video=="True":
            img_dst = cv2.resize(img, (int(224), 224)) #1280x960
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
            is_video = "True"   #"False"    
        
if __name__ == '__main__':
    main()  