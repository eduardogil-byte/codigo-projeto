import cv2
import numpy as np
import pandas as pd

# =========================
# DETECTAR PCB
# =========================
def detectar_pcb(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    lower_blue = np.array([90, 50, 50])
    upper_blue = np.array([140, 255, 255])

    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    kernel = np.ones((7,7), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None

    cnt = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(cnt)

    return img[y:y+h, x:x+w]


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
            "y": float(row["PosY"]),
            "package": row["Package"],
            "rot": float(row["Rot"])
        })

    return comps


# =========================
# PEGAR FIDUCIAIS
# =========================
def pegar_fiduciais(comps):
    return [c for c in comps if "FD" in c["ref"]]


# =========================
# TAMANHO POR PACKAGE (mm)
# =========================
def tamanho_por_package(package):
    tabela = {
        "0603": (1.6, 0.8),
        "0805": (2.0, 1.25),
        "SOT-23": (3.0, 1.5),
        "SOT223": (6.5, 3.5),
        "MLF32": (7.0, 7.0),
        "MSOP08": (3.0, 3.0),
    }

    for k in tabela:
        if k in package:
            return tabela[k]

    return (2.0, 2.0)


# =========================
# DESENHAR COMPONENTE
# =========================
def desenhar(img, x, y, comp, escala_px_por_mm, cor):
    w_mm, h_mm = tamanho_por_package(comp["package"])

    w = int(w_mm * escala_px_por_mm) + 10
    h = int(h_mm * escala_px_por_mm) + 10

    angle = -comp["rot"]

    rect = ((x, y), (w, h), angle)
    box = cv2.boxPoints(rect)
    box = np.int32(box)

    cv2.polylines(img, [box], True, cor, 2)

    cv2.putText(img, comp["ref"], (x-20, y-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, cor, 1)


# =========================
# CLICK
# =========================
pontos_img = []

def clicar(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        pontos_img.append([x, y])
        print("Clique:", x, y)


# =========================
# MAIN
# =========================
def main():
    img = cv2.imread("codigo-projeto/img2.jpeg")

    pcb = detectar_pcb(img)
    if pcb is None:
        pcb = img

    componentes = carregar_componentes("codigo-projeto/coordenadas_uno_r3_smd.csv")

    # pega fiduciais
    fid = pegar_fiduciais(componentes)

    print("Clique nos 3 fiduciais (FD1, FD2, FD3)")

    cv2.imshow("PCB", pcb)
    cv2.setMouseCallback("PCB", clicar)
    cv2.waitKey(0)

    if len(pontos_img) < 3:
        print("Precisa clicar 3 pontos")
        return

    pontos_img_np = np.array(pontos_img[:3], dtype=np.float32)

    pontos_csv = np.array([[f["x"], f["y"]] for f in fid[:3]], dtype=np.float32)

    # calcula homografia
    H, _ = cv2.findHomography(pontos_csv, pontos_img_np)

    debug = pcb.copy()

    # escala aproximada (usada só para tamanho)
    escala_px_por_mm = 5

    for comp in componentes:

        # ignorar coisas irrelevantes
        if "TP" in comp["ref"] or "FRAME" in comp["ref"]:
            continue

        pt = np.array([[comp["x"], comp["y"], 1]]).T
        dst = H @ pt

        x = int(dst[0] / dst[2])
        y = int(dst[1] / dst[2])

        x = max(0, min(x, pcb.shape[1]-1))
        y = max(0, min(y, pcb.shape[0]-1))

        desenhar(debug, x, y, comp, escala_px_por_mm, (0,255,0))

    cv2.imshow("RESULTADO", debug)
    cv2.waitKey(0)


if __name__ == "__main__":
    main()