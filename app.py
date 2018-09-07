from flask import Flask,jsonify,request
from flasgger import Swagger
from sklearn.externals import joblib
import json

bahasaID = Flask(__name__)
Swagger(bahasaID)

def createJSONSentenceList(listName):
    hasilKalimat = {}
    hasilKalimat['daftarKalimat'] = []    
    for index, item in enumerate(listName):
        list_kalimat = {}
        list_kalimat['no_kalimat'] = index
        list_kalimat['teks_kalimat'] = item
        hasilKalimat['daftarKalimat'].append(list_kalimat)
    return hasilKalimat

def createJSONWordsList(listName):
    hasilKata = {}
    hasilKata['daftarKata'] = []    
    for index, item in enumerate(listName):
        list_kata = {}
        list_kata['no_kata'] = index
        list_kata['teks_kata'] = item
        hasilKata['daftarKata'].append(list_kata)
    return hasilKata

@bahasaID.route('/sentTokenizer/doc', methods=['POST'])
def sent_tokenize():
    """
    Ini adalah endpoint untuk melakukan tokenisasi kalimat dari dokumen berbahasa Indonesia
    ---
    tags:
        - Rest Controller
    parameters:
        - name: body
          in: body
          required: true
          schema:
            id: Documents
            required:
                - document
            properties:
                document:
                    type: string
                    description: Silahkan isikan dokumen berbahasa Indonesia yang akan ditokenisasi ke dalam kalimat
                    default: ""
    responses:
        200:
            description: Berhasil
        400:
            description: Mohon maaf, ada permasalahan dalam memproses permintaan Anda

    """
    
    new_doc = request.get_json()
    doc = new_doc['document']
    

    tokenizer = joblib.load('bahasa-engine.pkl')
    resultToken = tokenizer[0](text=doc)
  
    return jsonify(createJSONSentenceList(resultToken))

@bahasaID.route('/wordTokenizer/sent', methods=['POST'])
def word_tokenize():
    """
    Ini adalah endpoint untuk melakukan tokenisasi kata dari kalimat berbahasa Indonesia
    ---
    tags:
        - Rest Controller
    parameters:
        - name: body
          in: body
          required: true
          schema:
            id: Sentence
            required:
                - sentence
            properties:
                sentence:
                    type: string
                    description: Silahkan isikan kalimat berbahasa Indonesia yang akan ditokenisasi ke dalam kalimat
                    default: ""
    responses:
        200:
            description: Berhasil
        400:
            description: Mohon maaf, ada permasalahan dalam memproses permintaan Anda

    """
    
    new_sent = request.get_json()
    sent = new_sent['sentence']
    

    tokenizer = joblib.load('bahasa-engine.pkl')
    resultToken = tokenizer[1](sent)
  
    return jsonify(createJSONWordsList(resultToken))

#bahasaID.run(debug=True)
