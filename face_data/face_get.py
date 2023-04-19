import cv2
import glob
import os

# メンバー名
member_name = "Ninomiya"

# 嵐メンバーの画像フォルダのパス
path ="/Volumes/GoogleDrive-116873471487367365175/マイドライブ/ArashiProject/" + member_name
# 嵐メンバーの画像フォルダの中の全画像のパスを取得して配列化
img_path_list=glob.glob(path + "/*")

# 番号の初期化
count = 0
number_face = 0
number_no_face = 0

# ディレクトリ名指定
dirname = member_name + "_face" 
# ディレクトリがない場合は作成
if not os.path.exists(dirname):
    os.mkdir(dirname)

# 画像パス配列から画像パスを取り出していくループ
for img_path in img_path_list:

    count += 1
    print(str(count) + "/" + str(len(img_path_list)))

    # 画像ファイル名を取得
    base_name = os.path.basename(img_path)
    # 画像ファイル名nameと拡張子extを取得
    name,ext = os.path.splitext(base_name)

    # 画像ファイル以外のファイルの場合はループをスキップ
    if (ext != '.jpg') and (ext != '.jpeg') and (ext != '.png') and (ext != '.PNG'):
        print("not a picture")
        continue

    if (ext == '.PNG'):
        ext = '.png'

    # 画像ファイルを読み込む
    img_src = cv2.imread(img_path, 1)
    # 画像をグレースケールへ変換
    img_gray = cv2.cvtColor(img_src, cv2.COLOR_RGB2GRAY)

    # カスケードファイルのパス
    cascade_path = "../haarcascade_frontalface_alt.xml"
    # カスケード分類器の特徴量取得
    cascade = cv2.CascadeClassifier(cascade_path)
    # 顔認識
    faces=cascade.detectMultiScale(img_gray, scaleFactor=1.1, minNeighbors=1, minSize=(40,40))

    # 顔がない場合はループをスキップ
    if len(faces) == 0:
        print("no face")
        number_no_face += 1
        # スキップ
        continue
    
    # 顔がある場合
    number_face += 1
    # 顔部分画像を取得
    for x,y,w,h in faces:
        face = img_src[y:y+h, x:x+w]

        # リサイズ
        face = cv2.resize(face, (64, 64))

        # 顔を検出できた画像を保存する
        
        # ファイル名指定
        file_name = dirname + "_" + str(number_face) + "_" + name + ext
        file_name = dirname + "_" + str(number_face) + "_" + ext
        # ディレクトリ名とファイル名を結合
        file_path = os.path.join(dirname, file_name)

        # ファイルの保存
        cv2.imwrite(file_path, face)