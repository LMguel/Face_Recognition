import os
import cv2
import csv
import datetime
from flask import Flask, request, jsonify

app = Flask(__name__)

KNOWN_FACES_DIR = "known_faces"
RECORDS_FILE = "records.csv"
os.makedirs(KNOWN_FACES_DIR, exist_ok=True)

# Garante que o arquivo CSV tem os cabeçalhos corretos
if not os.path.exists(RECORDS_FILE):
    with open(RECORDS_FILE, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["name", "date", "time"])

@app.route('/capture', methods=['POST'])
def capture_photo():
    """ Captura uma foto automaticamente e salva no sistema. """
    name = request.args.get('name')

    if not name:
        return jsonify({"error": "Nome do funcionário é obrigatório"}), 400

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        return jsonify({"error": "Não foi possível acessar a câmera"}), 500

    ret, frame = cap.read()
    cap.release()

    if not ret:
        return jsonify({"error": "Erro ao capturar imagem"}), 500

    file_path = os.path.join(KNOWN_FACES_DIR, f"{name}.jpg")
    cv2.imwrite(file_path, frame)

    return jsonify({"message": f"Foto de {name} cadastrada com sucesso!", "file": file_path}), 200

@app.route('/register', methods=['POST'])
def register_face():
    """ Registra um novo ponto enviando uma imagem. """
    if 'file' not in request.files:
        return jsonify({"error": "Nenhuma imagem enviada"}), 400

    file = request.files['file']
    name = os.path.splitext(file.filename)[0]

    now = datetime.datetime.now()
    date = now.strftime("%Y-%m-%d")
    time = now.strftime("%H:%M:%S")

    with open(RECORDS_FILE, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([name, date, time])

    return jsonify({"results": [{"name": name, "date": date, "time": time}]}), 200

@app.route('/records', methods=['GET'])
def get_records():
    """ Retorna os registros de ponto. """
    records = []
    try:
        with open(RECORDS_FILE, mode='r') as file:
            reader = csv.DictReader(file)
            for row in reader:
                records.append({
                    "name": row.get("name", "Nome não disponível"),
                    "date": row.get("date", "Data não disponível"),
                    "time": row.get("time", "Horário não disponível")
                })
    except Exception as e:
        return jsonify({"error": f"Erro ao ler registros: {str(e)}"}), 500

    return jsonify({"records": records}), 200

if __name__ == '__main__':
    app.run(debug=True)
