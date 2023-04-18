import pygame.mixer

def playMusic(predicted):
    Amus = "music/Aiba.mp3"
    Smus = "music/Sakurai.mp3"
    Nmus = "music/Ninomiya.mp3"
    Omus = "music/Ohno.mp3"
    Mmus = "music/Matsumoto.mp3"

    if(predicted==0): result = Amus
    elif(predicted==1): result = Mmus
    elif(predicted==2):result = Nmus
    elif(predicted==3): result = Omus
    else: result = Smus
    pygame.mixer.init() #初期化します
    pygame.mixer.music.load(result) #音声ファイルを読み込み
    pygame.mixer.music.play(1) #再生