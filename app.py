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
        list_kalimat['token_id'] = index
        list_kalimat['teks_kalimat'] = item
        hasilKalimat['daftarKalimat'].append(list_kalimat)
    return hasilKalimat

def createJSONWordsList(listName):
    hasilKata = {}
    hasilKata['daftarKata'] = []    
    for index, item in enumerate(listName):
        list_kata = {}
        list_kata['token_id'] = index
        list_kata['teks_kata'] = item
        hasilKata['daftarKata'].append(list_kata)
    return hasilKata

def createJSONStemsList(listName):
    hasilStem = {}
    hasilStem['daftarStem'] = []    
    for index, item in enumerate(listName):
        list_stem = {}
        list_stem['token_id'] = index
        list_stem['token_stem'] = item
        hasilStem['daftarStem'].append(list_stem)
    return hasilStem
def createJSONLemmaList(listName):
    hasilLemma = {}
    hasilLemma['daftarLemma'] = []    
    for token in listName:
        list_lemma = {}
        #list_lemma['token_text'] = token.text
        list_lemma['token_lemma'] = token.lemma_
        hasilLemma['daftarLemma'].append(list_lemma)
    return hasilLemma

@bahasaID.route('/segmentasi/doc', methods=['POST'])
def segmentasi():
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

    function = joblib.load('bahasa-engine.pkl')
    resultToken = function[0]['segmentasi'](text=doc)
  
    return jsonify(createJSONSentenceList(resultToken))

@bahasaID.route('/tokenisasi/sent', methods=['POST'])
def tokenisasi():
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

    function = joblib.load('bahasa-engine.pkl')
    resultToken = function[0]['tokenisasi'](sent)
  
    return jsonify(createJSONWordsList(resultToken))

@bahasaID.route('/stemming/sent', methods=['POST'])
def stemming():
    """
    Ini adalah endpoint untuk melakukan stemming  kalimat berbahasa Indonesia
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

    function = joblib.load('bahasa-engine.pkl')
    resultToken = function[0]['stemming'].stem(sent)
  
    return jsonify(createJSONStemsList(resultToken.split()))

@bahasaID.route('/lemmatisasi/sent', methods=['POST'])
def lemmatisasi():
    """
    Ini adalah endpoint untuk melakukan lemmatisasi kalimat berbahasa Indonesia
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
                    description: Silahkan isikan kalimat berbahasa Indonesia yang akan dilemmatizing ke dalam kalimat
                    default: ""
    responses:
        200:
            description: Berhasil
        400:
            description: Mohon maaf, ada permasalahan dalam memproses permintaan Anda

    """
    
    new_sent = request.get_json()
    sent = new_sent['sentence']

    function = joblib.load('../bahasa-engine.pkl')
    resultToken = function[0]['lemmatisasi'].lemma(sent)

    return jsonify(createJSONLemmaList(resultToken))
#bahasaID.run(debug=True)
