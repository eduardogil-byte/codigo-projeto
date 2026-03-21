import cv2
import numpy as np

# Função que encontra um retângulo grande na imagem e retorna apenas essa região
def detectar_pcb(caminho, varivale_extra=None):
    if varivale_extra is not None:
        img = varivale_extra
    else:
        img = cv2.imread(caminho)

    if img is None:
        print("Erro ao carregar imagem")
        return None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Suavizar
    blur = cv2.GaussianBlur(gray, (5,5), 0)

    # Bordas
    edges = cv2.Canny(blur, 50, 150)

    # Fechar gaps (IMPORTANTE)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7,7))
    
    # Dilata a image e Afina ao tamanho normal, para consertar caso estiver quebrada ou com
    # espacos vazios
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    # Contornos
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None

    # Ordena por área
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        # Ignora pequenos
        if area < 5000:
            continue

        #calcula o menor retangulo vertical que consegue envolver completamtnete o contorno
        x, y, w, h = cv2.boundingRect(cnt)

        # Filtrar por tamanho mínimo
        if w < 50 or h < 50:
            continue

        #  Filtrar por proporção (retangular)
        proporcao = w / float(h)

        if 0.5 < proporcao < 5:  # bem mais flexível
            x, y, w, h = cv2.boundingRect(cnt)

            roi = img[y:y+h, x:x+w]

            # # DEBUG
            # cv2.imshow("edges", edges)

            # img_debug = img.copy()
            # cv2.drawContours(img_debug, [cnt], -1, (0,255,0), 3)
            # cv2.imshow("contorno PCB", img_debug)

            return roi

    return None


def redimensionar_imagem(img, nova_largura=1024):
    # quero uma largura e vou dividir com a largura da img
    proporcao = float(nova_largura) / img.shape[1] 
    # aqui pegar a altura anterior e multiplicamos pela proporcao
    nova_altura = int(img.shape[0] * proporcao)

    nova_img = cv2.resize(img, (nova_largura,nova_altura), interpolation=cv2.INTER_LINEAR)

    cv2.imwrite("redimensionada.jpg", nova_img)

    return nova_img



def comparar_pcb(img_referencia, img_teste, tamanho=30, limiar_dif=50):
    """
    img_referencia: imagem da PCB perfeita
    img_teste: imagem da PCB a verificar
    tamanho: tamanho do quadrado para verificar pequenas regiões
    limiar_dif: diferença mínima para considerar componente ausente
    """
    debug = img_teste.copy()
    
    # Garantir que as imagens tenham o mesmo tamanho
    if img_referencia.shape != img_teste.shape:
        img_teste = cv2.resize(img_teste, (img_referencia.shape[1], img_referencia.shape[0]))

    h, w = img_referencia.shape[:2]

    # Percorrer a imagem em blocos
    for y in range(0, h, tamanho):
        for x in range(0, w, tamanho):
            # Recortes
            ref_roi = img_referencia[y:y+tamanho, x:x+tamanho]
            test_roi = img_teste[y:y+tamanho, x:x+tamanho]

            # Evitar bordas incompletas
            if ref_roi.shape[0] == 0 or ref_roi.shape[1] == 0:
                continue

            # Converter para cinza
            ref_gray = cv2.cvtColor(ref_roi, cv2.COLOR_BGR2GRAY)
            test_gray = cv2.cvtColor(test_roi, cv2.COLOR_BGR2GRAY)

            # Diferença absoluta
            diff = cv2.absdiff(ref_gray, test_gray)
            _, diff_bin = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
            soma_dif = np.sum(diff_bin)

            # Se a diferença for grande → componente ausente
            if soma_dif > limiar_dif:
                # Marcar em vermelho
                cv2.rectangle(debug, (x, y), (x+tamanho, y+tamanho), (0,0,255), 1)
            else:
                # Marcar em verde (opcional)
                cv2.rectangle(debug, (x, y), (x+tamanho, y+tamanho), (0,255,0), 1)

    return debug

