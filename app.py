from flask import Flask, request, jsonify
import cv2
import os
import face_recognition
import numpy as np
import datetime
import pickle

app = Flask(__name__)

# Diretórios
IMAGE_DIR = "capturas"
ENCODINGS_FILE = "face_encodings.pkl"

# Criar diretórios se não existirem
if not os.path.exists(IMAGE_DIR):
    os.makedirs(IMAGE_DIR)

# Carregar dados salvos
if os.path.exists(ENCODINGS_FILE):
    with open(ENCODINGS_FILE, "rb") as f:
        known_faces = pickle.load(f)
else:
    known_faces = {}

# 📌 1️⃣ Cadastro de funcionário (salvar rosto)
@app.route('/register', methods=['POST'])
def register():
    """ Captura a foto do funcionário e armazena a assinatura facial """
    data = request.get_json()

    if not data or "nome" not in data:
        return jsonify({"error": "Nome do funcionário é obrigatório"}), 400

    nome = data["nome"]
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{nome}_{timestamp}.jpg"
    filepath = os.path.join(IMAGE_DIR, filename)

    # Captura a imagem da webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        return jsonify({"error": "Erro ao acessar a webcam"}), 500

    ret, frame = cap.read()
    cap.release()

    if not ret:
        return jsonify({"error": "Erro ao capturar imagem"}), 500

    # Salvar a imagem
    cv2.imwrite(filepath, frame)

    # Analisar a imagem para extrair o rosto
    face_encoding = extract_face_encoding(filepath)
    if face_encoding is None:
        return jsonify({"error": "Nenhum rosto detectado"}), 400

    # Armazena no banco de dados (dicionário + arquivo)
    known_faces[nome] = face_encoding
    with open(ENCODINGS_FILE, "wb") as f:
        pickle.dump(known_faces, f)

    return jsonify({"message": f"Funcionário {nome} cadastrado com sucesso!", "file": filename})


# 📌 2️⃣ Reconhecimento facial e registro de ponto
@app.route('/recognize', methods=['POST'])
def recognize():
    """ Captura uma nova foto e verifica se o rosto é reconhecido """
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        return jsonify({"error": "Erro ao acessar a webcam"}), 500

    ret, frame = cap.read()
    cap.release()

    if not ret:
        return jsonify({"error": "Erro ao capturar imagem"}), 500

    # Salvar temporariamente
    temp_filepath = os.path.join(IMAGE_DIR, "temp.jpg")
    cv2.imwrite(temp_filepath, frame)

    # Extrair o rosto da nova foto
    unknown_encoding = extract_face_encoding(temp_filepath)
    if unknown_encoding is None:
        return jsonify({"error": "Nenhum rosto detectado"}), 400

    # Comparar com os rostos cadastrados
    for nome, known_encoding in known_faces.items():
        results = face_recognition.compare_faces([known_encoding], unknown_encoding)
        if results[0]:  # Se houver correspondência
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            return jsonify({"message": f"Registro de ponto bem-sucedido!", "funcionario": nome, "horario": timestamp})

    return jsonify({"error": "Rosto não reconhecido"}), 404


# 📌 Função auxiliar: Extrair assinatura facial
def extract_face_encoding(image_path):
    """ Lê a imagem e retorna a assinatura facial (encoding) """
    image = face_recognition.load_image_file(image_path)
    encodings = face_recognition.face_encodings(image)
    if len(encodings) > 0:
        return encodings[0]  # Retorna apenas o primeiro rosto encontrado
    return None


@app.route('/', methods=['GET'])
def home():
    return "API de Registro de Ponto Eletrônico com Reconhecimento Facial", 200


if __name__ == '__main__':
    app.run(debug=True)
