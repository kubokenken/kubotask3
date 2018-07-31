【プログラム】
```java
import cv2
import numpy as np

def myfunc(x):
    pass

def skin_detect(img):
    # HSV色空間に変換
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # 赤色のHSVの値域1
    hsv_min = np.array([0,40,0])
    hsv_max = np.array([40,255,255])
    mask1 = cv2.inRange(hsv, hsv_min, hsv_max)

    return mask1


def main():
    cv2.namedWindow('Frame1')
    cv2.createTrackbar('R', 'Frame1', 0, 255, myfunc)
    cv2.createTrackbar('G', 'Frame1', 0, 255, myfunc)
    cv2.createTrackbar('B', 'Frame1', 0, 255, myfunc)
    # カメラのキャプチャ
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    img = np.zeros((300,512,3),np.uint8)

    while(cap.isOpened()):
        # フレームを取得
        ret, frame = cap.read()

        # 赤色検出
        median = cv2.medianBlur(frame,5) #中央値フィルタ
        mask = skin_detect(median)
        # ORB (Oriented FAST and Rotated BRIEF)
        detector = cv2.ORB_create()
        res = cv2.bitwise_and(frame,frame, mask= mask)
        r = cv2.getTrackbarPos('R', 'Frame1')
        g = cv2.getTrackbarPos('G', 'Frame1')
        b = cv2.getTrackbarPos('B', 'Frame1')
        frame1 = frame - res
#        res1 = cv2.cvtColor(res, cv2.COLOR_HSV2BGR)
        res1 = res.copy()
        res_B = res.copy()
        res_B[:,:,(1,2)] = 0
        res_G = res.copy()
        res_G[:,:,(0,2)] = 0
        res_R = res.copy()
        res_R[:,:,(0,1)] = 0
        res1 = res_R * r / 255 + res_G * g / 255 + res_B * b / 255

        img[:] = [b,g,r]

        frame2 = frame1 + res1

        # 結果表示
        cv2.imshow('image',img)
        cv2.imshow("Frame", frame1)
        cv2.imshow("Frame1", frame2)
        cv2.imshow("Mask", mask)
#        plt.colorbar(mask)
        # qキーが押されたら途中終了
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()

```

以上のプログラムを以下のコードで実行する.
C:\Users\owner\github>python cam_parts.py

---

【説明】
skin_detect関数でいろ追跡をする.今回は肌を検出するのでRBG値を[0,40,0]から[40,255,255]とする.
21から30行目でカメラのキャプチャとカラーバーの設定.
37から58でskin_detect関数を使って検出した肌部分の色を変更する処理をする.
カラーバーで変色して出来た肌の部分をres1に入れて下の画像からresを引いた画像のframe1と合成することで肌の色が変わるframe2ができる.
ここでframe1,frame2としているが表示するときはそれぞれFrame,Frame1としている.
61から65で処理したカメラ画像を表示している.
67行目はqを押すとカメラを停止する処理が書かれている.

【動画URL】
https://www.youtube.com/watch?v=EBW-eLqobNg&feature=youtu.be
