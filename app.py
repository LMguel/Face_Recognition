import os
import json
import cv2
import numpy as np
import face_recognition
from flask import Flask, request, jsonify
from flask_cors import CORS
from datetime import datetime

app = Flask(__name__)
CORS(app)

# Caminhos
FACE_PATH = "faces"  # Pasta onde as imagens registradas ficarão salvas
REGISTROS_PATH = "registros.json"

# Se o arquivo de registros não existir, cria um vazio
if not os.path.exists(REGISTROS_PATH):
    with open(REGISTROS_PATH, "w") as f:
        json.dump([], f)

# Se a pasta de rostos não existir, cria
if not os.path.exists(FACE_PATH):
    os.makedirs(FACE_PATH)


def encode_faces():
    """Carrega e codifica todas as imagens registradas."""
    known_face_encodings = []
    known_face_names = []

    for filename in os.listdir(FACE_PATH):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(FACE_PATH, filename)
            image = face_recognition.load_image_file(image_path)
            encodings = face_recognition.face_encodings(image)

            if encodings:
                known_face_encodings.append(encodings[0])
                known_face_names.append(filename.split("_")[0])  # Usa o nome do arquivo como nome do funcionário

    return known_face_encodings, known_face_names


@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "API rodando!"}), 200


@app.route("/register", methods=["POST"])
def register():
    """Captura uma imagem da câmera e cadastra um funcionário."""
    data = request.get_json()
    nome = data.get("nome")

    if not nome:
        return jsonify({"error": "Nome é obrigatório"}), 400

    # Captura a imagem da câmera
    cam = cv2.VideoCapture(0)  # 0 para webcam principal
    ret, frame = cam.read()
    cam.release()

    if not ret:
        return jsonify({"error": "Erro ao acessar a câmera"}), 500

    filename = f"{nome}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
    filepath = os.path.join(FACE_PATH, filename)

    cv2.imwrite(filepath, frame)  # Salva a imagem

    return jsonify({
        "file": filename,
        "message": f"Funcionário {nome} cadastrado com sucesso!"
    })


@app.route("/recognize", methods=["POST"])
def recognize():
    """Captura uma imagem e verifica se o rosto pertence a alguém cadastrado."""
    known_encodings, known_names = encode_faces()

    if not known_encodings:
        return jsonify({"error": "Nenhum rosto cadastrado!"}), 400

    # Captura a imagem da câmera
    cam = cv2.VideoCapture(0)
    ret, frame = cam.read()
    cam.release()

    if not ret:
        return jsonify({"error": "Erro ao acessar a câmera"}), 500

    # Converte a imagem para RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detecta rostos na imagem capturada
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    for face_encoding in face_encodings:
        # Compara com os rostos conhecidos
        matches = face_recognition.compare_faces(known_encodings, face_encoding)
        name = "Desconhecido"

        if True in matches:
            first_match_index = matches.index(True)
            name = known_names[first_match_index]

            # Salvar registro de ponto
            horario = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            novo_registro = {"funcionario": name, "horario": horario}

            with open(REGISTROS_PATH, "r") as f:
                registros = json.load(f)

            registros.append(novo_registro)

            with open(REGISTROS_PATH, "w") as f:
                json.dump(registros, f, indent=4)

            return jsonify({
                "funcionario": name,
                "horario": horario,
                "message": "Registro de ponto bem-sucedido!"
            })

    return jsonify({"error": "Rosto não reconhecido"}), 400


@app.route("/records", methods=["GET"])
def get_records():
    """Retorna todos os registros de ponto."""
    with open(REGISTROS_PATH, "r") as f:
        registros = json.load(f)

    return jsonify({"registros": registros})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
