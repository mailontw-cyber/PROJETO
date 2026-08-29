import cv2
import numpy as np
import os


def aplicar_processamentos(caminho_imagem):
    imagem = cv2.imread(caminho_imagem)

    if imagem is None:
        raise ValueError("Não foi possível abrir a imagem.")

    os.makedirs("processadas", exist_ok=True)

    nome_arquivo = os.path.basename(caminho_imagem)
    nome = os.path.splitext(nome_arquivo)[0]

    resultados = {}

    # EROSÃO
    kernel = np.ones((3, 3), np.uint8)

    erosao = cv2.erode(imagem, kernel, iterations=1)
    caminho_erosao = f"processadas/{nome}_erosao.jpg"
    cv2.imwrite(caminho_erosao, erosao)
    resultados["erosao"] = caminho_erosao

    # DILATAÇÃO
    dilatacao = cv2.dilate(imagem, kernel, iterations=1)
    caminho_dilatacao = f"processadas/{nome}_dilatacao.jpg"
    cv2.imwrite(caminho_dilatacao, dilatacao)
    resultados["dilatacao"] = caminho_dilatacao

    # GAUSSIANO
    gaussiano = cv2.GaussianBlur(imagem, (5, 5), 0)
    caminho_gaussiano = f"processadas/{nome}_gaussiano.jpg"
    cv2.imwrite(caminho_gaussiano, gaussiano)
    resultados["gaussiano"] = caminho_gaussiano

    # GRAYSCALE
    grayscale = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
    caminho_grayscale = f"processadas/{nome}_grayscale.jpg"
    cv2.imwrite(caminho_grayscale, grayscale)
    resultados["grayscale"] = caminho_grayscale

    # SOBEL
    sobel_x = cv2.Sobel(grayscale, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(grayscale, cv2.CV_64F, 0, 1, ksize=3)

    sobel_x = cv2.convertScaleAbs(sobel_x)
    sobel_y = cv2.convertScaleAbs(sobel_y)

    sobel = cv2.addWeighted(sobel_x, 0.5, sobel_y, 0.5, 0)

    caminho_sobel = f"processadas/{nome}_sobel.jpg"
    cv2.imwrite(caminho_sobel, sobel)
    resultados["sobel"] = caminho_sobel

    return resultados