import cv2
import numpy as np

def generate_adaptive_circular_contour(image, num_points=100, margin=1.1):
    """
    Gera um contorno circular adaptado à lesão, com base em Otsu + bounding box.

    Retorna:
      - Contorno circular (num_points, 2)
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thresh = cv2.bitwise_not(thresh)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        print("Fallback pra circulo central...")
        return generate_circular_contour(image.shape, radius_ratio=0.2, num_points=num_points)

    largest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)

    center = (x + w / 2, y + h / 2)
    a = (w / 2) * margin  # semi-eixo x
    b = (h / 2) * margin  # semi-eixo y

    angles = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
    xs = center[0] + a * np.cos(angles)
    ys = center[1] + b * np.sin(angles)

    contour = np.stack([xs, ys], axis=1).astype(np.float32)
    return contour

def generate_circular_contour(image_shape, radius_ratio=0.2, num_points=100):
    """
    Gera um contorno circular centralizado na imagem com raio proporcional ao tamanho.

    Parâmetros:
      - image_shape: (h, w) da imagem
      - radius_ratio: proporção do menor lado da imagem (ex: 0.2 = 20%)
      - num_points: número de pontos da curva

    Retorna:
      - contour: array (num_points, 2) float32
    """
    h, w = image_shape[:2]
    center = (w / 2, h / 2)
    radius = min(h, w) * radius_ratio

    angles = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
    x = center[0] + radius * np.cos(angles)
    y = center[1] + radius * np.sin(angles)

    contour = np.stack([x, y], axis=1).astype(np.float32)
    return contour


def compute_gvf(image, mu=0.2, iterations=80, delta_t=1):
    """
    Calcula o fluxo de vetores de gradiente (GVF) para a imagem.

    Parâmetros:
      - image: imagem de entrada (RGB ou escala de cinza);
      - mu: parâmetro de regularização que controla o equilíbrio entre suavização e fidelidade às bordas;
      - iterations: número de iterações para a evolução do campo;
      - delta_t: passo temporal para cada iteração.

    Retorna:
      - u, v: componentes do campo GVF.
    """
    # Converte para escala de cinza se necessário
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image.copy()
    gray = gray.astype(np.float32) / 255.0

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray.astype(np.uint8))

    # Calcula os gradientes com Sobel
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=5)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=5)
    grad_mag = np.sqrt(gx ** 2 + gy ** 2)

    # Função de borda não-linear
    f = cv2.GaussianBlur(grad_mag, (5, 5), 0)

    # Gradientes do mapa de bordas
    f_grad_x = cv2.Sobel(f, cv2.CV_32F, 1, 0, ksize=5)
    f_grad_y = cv2.Sobel(f, cv2.CV_32F, 0, 1, ksize=5)

    # Inicializa o GVF com os gradientes do mapa de bordas
    u = f_grad_x.copy()
    v = f_grad_y.copy()

    # Calcula e normaliza o termo de penalização para evitar overflow
    squared_mag = f_grad_x ** 2 + f_grad_y ** 2
    squared_mag = squared_mag / (np.max(squared_mag) + 1e-8)

    for i in range(iterations):
        u_lap = cv2.Laplacian(u, cv2.CV_32F)
        v_lap = cv2.Laplacian(v, cv2.CV_32F)

        u += delta_t * (mu * u_lap - (u - f_grad_x) * squared_mag)
        v += delta_t * (mu * v_lap - (v - f_grad_y) * squared_mag)

    # Previne que valores inválidos se propaguem
    u = np.nan_to_num(u, nan=0.0, posinf=1.0, neginf=-1.0)
    v = np.nan_to_num(v, nan=0.0, posinf=1.0, neginf=-1.0)

    return u, v


def interp_gvf_vectorized(contour, u, v):
    """
    Realiza a interpolação bilinear do campo GVF de maneira vetorizada para
    um conjunto de pontos.
    """
    h, w = u.shape
    # Separa as coordenadas e garante que fiquem dentro dos limites para interpolação
    x = np.clip(contour[:, 0], 0, w - 2)
    y = np.clip(contour[:, 1], 0, h - 2)

    x0 = np.floor(x).astype(np.int32)
    y0 = np.floor(y).astype(np.int32)
    x1 = np.minimum(x0 + 1, w - 1)
    y1 = np.minimum(y0 + 1, h - 1)

    dx = x - x0
    dy = y - y0

    # Calcula a interpolação vetorizada para ambos os componentes
    u_interp = (u[y0, x0] * (1 - dx) * (1 - dy) +
                u[y0, x1] * dx * (1 - dy) +
                u[y1, x0] * (1 - dx) * dy +
                u[y1, x1] * dx * dy)
    v_interp = (v[y0, x0] * (1 - dx) * (1 - dy) +
                v[y0, x1] * dx * (1 - dy) +
                v[y1, x0] * (1 - dx) * dy +
                v[y1, x1] * dx * dy)

    return np.column_stack([u_interp, v_interp])

def gvf_based_segmentation(image, init_contour, mu=0.2, iterations=80, delta_t=1,
                           alpha=0.1, beta=0.1, gamma=1, kappa=0.5, iterations_snake=100):
    """
    Realiza segmentação baseada em GVF aplicando um contorno ativo (snake).

    Parâmetros:
      - image: imagem de entrada (RGB ou escala de cinza);
      - init_contour: numpy.array de formato (N, 2) com os pontos iniciais da snake (ex: contorno inicial);
      - mu, iterations, delta_t: parâmetros para cálculo do GVF;
      - alpha: peso relacionado à tensão (energia de contorno);
      - beta: peso relacionado à rigidez (energia de flexão);
      - gamma: passo que controla a contribuição da energia interna;
      - kappa: peso da força externa (do campo GVF);
      - iterations_snake: número de iterações para evolução da snake.

    Retorna:
      - contour: contorno final (numpy.array com formato (N, 2)).
    """
    u, v = compute_gvf(image, mu=mu, iterations=iterations, delta_t=delta_t)

    # Inicializa o contorno
    contour = init_contour.copy()
    h, w = u.shape

    for it in range(iterations_snake):
        # Obtem os vizinhos usando np.roll (contorno fechado)
        prev = np.roll(contour, 1, axis=0)
        next = np.roll(contour, -1, axis=0)

        # Energia interna combinada de tensão e rigidez de forma vetorizada
        tension = alpha * (next - 2 * contour + prev)
        rigidity = beta * ((next - contour) - (contour - prev))
        internal_force = tension + rigidity

        # Energia externa: interpola o campo GVF de forma vetorizada para todos os pontos
        external_force = interp_gvf_vectorized(contour, u, v)

        # Atualiza o contorno de forma vetorizada
        contour += gamma * internal_force + kappa * external_force

        # Garante que os pontos permaneçam dentro dos limites da imagem
        contour[:, 0] = np.clip(contour[:, 0], 0, w - 1)
        contour[:, 1] = np.clip(contour[:, 1], 0, h - 1)

    return contour