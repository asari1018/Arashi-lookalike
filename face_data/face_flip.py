import cv2
import glob
import os

# メンバー名
member_name = "Ninomiya"

# 嵐メンバーの画像フォルダのパス
path ="/Volumes/GoogleDrive-116873471487367365175/マイドライブ/ArashiProject/" + member_name + "_face"
# 嵐メンバーの画像フォルダの中の全画像のパスを取得して配列化
img_path_list=glob.glob(path + "/*")

# 番号の初期化
count = 0

# 画像パス配列から画像パスを取り出していくループ
for img_path in img_path_list:

    count += 1
    print(str(count) + "/" + str(len(img_path_list)))

    # 画像ファイル名を取得
    base_name = os.path.basename(img_path)
    # 画像ファイル名nameと拡張子extを取得
    name,ext = os.path.splitext(base_name)

    # 画像ファイル以外のファイルの場合はループをスキップ
    if (ext != '.jpg') and (ext != '.jpeg') and (ext != '.png'):
        print("not a picture")
        continue

    # 画像ファイルを読み込む
    img_src = cv2.imread(img_path, 1)

    img_flip = cv2.flip(img_src, 1)

    # ディレクトリ名指定
    dirname = member_name + "_face" 
    # ディレクトリがない場合は作成
    if not os.path.exists(dirname):
        os.mkdir(dirname)
    
    # ファイル名指定
    file_name = dirname + "_flip_" + str(count) + "_" + ext
    # ディレクトリ名とファイル名を結合
    file_path = os.path.join(dirname, file_name)

    # ファイルの保存
    cv2.imwrite(file_path, img_flip)