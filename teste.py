import cv2
import numpy as np
import pandas as pd

# =========================
# DETECTAR PCB (POR COR)
# =========================
def detectar_pcb(caminho, varivale_extra=None):
    if varivale_extra is not None:
        img = varivale_extra
    else:
        img = cv2.imread(caminho)

    if img is None:
        print("Erro ao carregar imagem")
        return None

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # azul da PCB
    lower_blue = np.array([90, 50, 50])
    upper_blue = np.array([140, 255, 255])

    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    kernel = np.ones((7,7), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        print("Nenhum contorno encontrado")
        return None

    cnt = max(contours, key=cv2.contourArea)

    x, y, w, h = cv2.boundingRect(cnt)

    pcb = img[y:y+h, x:x+w]

    # DEBUG visual
    debug = img.copy()
    cv2.rectangle(debug, (x,y), (x+w,y+h), (0,255,0), 3)
    cv2.imshow("DETECCAO PCB", debug)

    return pcb


# =========================
# REDIMENSIONAR
# =========================
def redimensionar_imagem(img, nova_largura=1024):
    proporcao = float(nova_largura) / img.shape[1]
    nova_altura = int(img.shape[0] * proporcao)
    return cv2.resize(img, (nova_largura, nova_altura))


# =========================
# CSV
# =========================
def carregar_componentes(csv_path):
    df = pd.read_csv(csv_path)

    comps = []
    for _, row in df.iterrows():
        comps.append({
            "ref": row["Ref"],
            "x": float(row["PosX"]),
            "y": float(row["PosY"])
        })

    return comps


# =========================
# ROI
# =========================
def extrair_roi(img, comp, escala_x, escala_y, offset_x, offset_y, tamanho=20):
    x = int(comp["x"] * escala_x + offset_x)
    y = int(comp["y"] * escala_y + offset_y)

    if x < 0 or y < 0 or x >= img.shape[1] or y >= img.shape[0]:
        return None, x, y

    roi = img[max(0, y-tamanho):y+tamanho, max(0, x-tamanho):x+tamanho]

    return roi, x, y


# =========================
# DETECÇÃO DE PRESENÇA
# =========================
def tem_componente(roi, limiar=20):
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    variacao = np.std(gray)
    return variacao > limiar


# =========================
# MAIN
# =========================
def main():
    img = cv2.imread("img.jpg")

    if img is None:
        print("Erro ao carregar imagem")
        return

    # Detecta PCB
    pcb = detectar_pcb(None, img)

    if pcb is None:
        print("Nao encontrou PCB, usando imagem inteira")
        pcb = img

    pcb = redimensionar_imagem(pcb)

    # CSV
    componentes = carregar_componentes("coordenadas.csv")

    # =========================
    # NORMALIZAÇÃO
    # =========================
    xs = [c["x"] for c in componentes]
    ys = [c["y"] for c in componentes]

    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    h, w = pcb.shape[:2]

    escala_x = w / (max_x - min_x)
    escala_y = h / (max_y - min_y)

    offset_x = -min_x * escala_x
    offset_y = -min_y * escala_y

    print("Escala:", escala_x, escala_y)
    print("Offset:", offset_x, offset_y)

    # =========================
    # DEBUG FINAL
    # =========================
    debug = pcb.copy()

    for comp in componentes[:200]:
        roi, x, y = extrair_roi(
            pcb, comp,
            escala_x, escala_y,
            offset_x, offset_y
        )

        if roi is None or roi.size == 0:
            continue

        existe = tem_componente(roi)

        cor = (0,255,0) if existe else (0,0,255)

        # quadrado
        cv2.rectangle(debug, (x-20, y-20), (x+20, y+20), cor, 2)

        # texto (REF)
        cv2.putText(
            debug,
            comp["ref"],
            (x - 20, y - 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            cor,
            1,
            cv2.LINE_AA
        )

    cv2.imshow("PCB", pcb)
    cv2.imshow("RESULTADO", debug)
    cv2.waitKey(0)


if __name__ == "__main__":
    main()