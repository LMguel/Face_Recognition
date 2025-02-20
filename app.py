from flask import Flask, request, jsonify
import face_recognition
import cv2
import numpy as np
import os
import datetime

app = Flask(__name__)

# Listas para armazenar os encodings e os nomes dos funcionários conhecidos.
known_face_encodings = []
known_face_names = []

def load_known_faces(directory="known_faces"):
    """
    Carrega imagens dos funcionários conhecidos, calcula seus encodings e armazena
    os resultados em listas.
    """
    for filename in os.listdir(directory):
        if filename.lower().endswith((".jpg", ".png", ".jpeg")):
            image_path = os.path.join(directory, filename)
            image = face_recognition.load_image_file(image_path)
            encodings = face_recognition.face_encodings(image)
            if encodings:
                known_face_encodings.append(encodings[0])
                # O nome do funcionário é extraído do nome do arquivo (sem extensão)
                name = os.path.splitext(filename)[0]
                known_face_names.append(name)
    print(f"Carregado {len(known_face_names)} funcionários conhecidos.")

# Carregar os dados dos funcionários conhecidos assim que a API iniciar.
load_known_faces()

# Rota raiz para informar que a API está rodando
@app.route('/')
def home():
    return jsonify({"message": "API de registro de ponto está rodando. Use o endpoint /register para enviar uma imagem."}), 200

@app.route('/register', methods=['POST'])
def register_attendance():
    """
    Endpoint para registrar o ponto. Espera um arquivo de imagem enviado via POST.
    O fluxo é:
      1. Receber a imagem.
      2. Converter a imagem para um array do OpenCV.
      3. Detectar faces e calcular encodings.
      4. Comparar com os encodings conhecidos.
      5. Retornar o(s) nome(s) detectado(s) e o timestamp.
    """
    if 'file' not in request.files:
        return jsonify({"error": "Nenhum arquivo enviado."}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "Nenhum arquivo selecionado."}), 400

    # Lê o arquivo e converte para array numpy
    file_bytes = file.read()
    np_arr = np.frombuffer(file_bytes, np.uint8)
    image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Detecta as localizações e os encodings das faces na imagem
    face_locations = face_recognition.face_locations(rgb_image)
    face_encodings = face_recognition.face_encodings(rgb_image, face_locations)

    results = []
    timestamp = datetime.datetime.now().isoformat()

    for face_encoding in face_encodings:
        # Compara o encoding detectado com os conhecidos
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Desconhecido"

        # Calcula as distâncias para encontrar a melhor correspondência
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]

        # Aqui você pode registrar a presença (ex: salvar em banco de dados)
        results.append({
            "name": name,
            "timestamp": timestamp
        })

    return jsonify({"results": results}), 200

if __name__ == '__main__':
    # Execute o app em debug durante o desenvolvimento.
    app.run(debug=True)
