import pandas as pd
import io
import os
from sys import stdin,stdout,stderr
import numpy as np
import itertools
from time import sleep
from math import sqrt

from sklearn.externals import joblib
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA

from skimage.feature import canny
from skimage.filters import sobel
from skimage.morphology import convex_hull_image

clf_figures = joblib.load('models/MLP_Figure_recognizer.joblib');
clf_color = joblib.load('models/LogReg_Color_recognizer.joblib')
pca = joblib.load('models/pca.joblib')
colpca = joblib.load('models/colpca50.joblib')

prevBoard = 0
prevBoardAug = 0

def data_aug_fast(data):
    global prevBoard
    global prevBoardAug
    images = np.zeros_like(data)
    chess_list_extended = np.zeros((data.shape[0],data.shape[1]*2,data.shape[2]))
    for pos,item in enumerate(data):
        if(isinstance(prevBoard,int)==False and np.all(prevBoard[pos]==data[pos])):
            chess_list_extended[pos] = prevBoardAug[pos]
            continue
        
        item = sobel(item)
        images[pos][3:24,3:24] = item[3:24,3:24]
        k = 5
        l = 0.001
        r = 3
        while(abs(r-l)>0.1):
            k = (l+r)/2.0
            img = canny(images[pos],k)
            s = sum(sum(img))
            if(s>256):
                l = k
            else:
                r = k
        images[pos] = canny(images[pos],l)
        images[pos] = convex_hull_image(images[pos] == 1)

        k = 0.04
        l = 0.04
        r = 0.2
        while(abs(r-l)>0.1):
            k = (l+r)/2.0
            img = item>k
            s = sum(sum(img))
            if(s<2000):
                r = k
            else:
                l = k
        chess_list_extended[pos] = list(images[pos]) + list(item>l)
    prevBoardAug = chess_list_extended.copy()
    return pd.DataFrame(chess_list_extended.reshape(chess_list_extended.shape[0],
                                                    chess_list_extended.shape[1]*chess_list_extended.shape[2]))

def isOpen(f):
    try:
        os.rename(f, f)
    except OSError as e:
        return True
    return False

def makeFen(board, Col):
    board/=255.0
    tempBoard = board.copy()
    global prevBoard

    board = data_aug_fast(board)
    board = pca.transform(board)
    
    prevBoard = tempBoard.copy()

    B = clf_figures.predict(board).reshape(8,8)
    Type = B.reshape(64,-1)
    board2 = pd.DataFrame(tempBoard.reshape(64,-1))
    
    board2 = board2[Type!='empty']
    if(board2.shape[0]==0):
        return '8/8/8/8/8/8/8/8 w - - 0 1'
    board2 = colpca.transform(board2)
    
    col = list(reversed(clf_color.predict(board2))) if Col=='b' else clf_color.predict(board2)
    fen = []
    k = 0
    for i in range(0,8) if Col=='w' else reversed(range(0,8)):
        br=0
        for j in range(0,8) if Col=='w' else reversed(range(0,8)):
            if B[i,j]=='empty':
                br+=1
            elif B[i,j]=='pawn':
                if(br>0):
                    fen.append(str(br))
                    br = 0
                fen.append('p' if col[k]=='black' else 'P')
                k+=1
            elif B[i,j]=='rook':
                if(br>0):
                    fen.append(str(br))
                    br = 0
                fen.append('r' if col[k]=='black' else 'R')
                k+=1
            elif B[i,j]=='knight':
                if(br>0):
                    fen.append(str(br))
                    br = 0
                fen.append('n' if col[k]=='black' else 'N')
                k+=1
            elif B[i,j]=='bishop':
                if(br>0):
                    fen.append(str(br))
                    br = 0
                fen.append('b' if col[k]=='black' else 'B')
                k+=1
            elif B[i,j]=='queen':
                if(br>0):
                    fen.append(str(br))
                    br = 0
                fen.append('q' if col[k]=='black' else 'Q')
                k+=1
            elif B[i,j]=='king':
                if(br>0):
                    fen.append(str(br))
                    br = 0
                fen.append('k' if col[k]=='black' else 'K')
                k+=1
        if(br>0):
            fen.append(str(br))
        if((Col=='w' and i!=7) or (Col=='b' and i!=0)):
            fen.append('/')
    finish = ' {} - - 0 1'.format(Col)
    return ((''.join(fen))+finish)
    
