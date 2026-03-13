"""
FraudGuard — Web Server

Run:   python3 app.py
Open:  http://localhost:8000

Requires: pip install flask scikit-learn pandas numpy
Optional: pip install fastapi uvicorn  (see app_fastapi.py)
"""
import os, json, math, pickle
import numpy as np
from flask import Flask, request, jsonify, send_from_directory, Response

BASE = os.path.dirname(os.path.abspath(__file__))
app  = Flask(__name__)

MERCHANT_CATS = ['entertainment','gas','grocery','online','restaurant','retail','travel']
FEAT_NAMES    = ['Amount','Hour','Day of Week','Merchant Cat',
                 'Dist. Home','Dist. Last Txn','Ratio to Median',
                 'PIN Used','Online Order','Repeat Retailer','Account Age',
                 'Txns 24h','Txns 7d','V1','V2','V3','V4','V5']
MEAN_VALS = [250,12,3,3,15,10,1.0,0.7,0.2,0.8,730,3,15,0,0,0,0,0]

_pkg = _analysis = None

def get_model():
    global _pkg
    if _pkg is None:
        with open(os.path.join(BASE,'models','best_model.pkl'),'rb') as f: _pkg = pickle.load(f)
    return _pkg

def get_analysis():
    global _analysis
    if _analysis is None:
        with open(os.path.join(BASE,'static','analysis.json')) as f: _analysis = json.load(f)
    return _analysis

def cj(obj):
    if isinstance(obj, float): return 0.0 if (math.isnan(obj) or math.isinf(obj)) else obj
    if isinstance(obj, dict):  return {k:cj(v) for k,v in obj.items()}
    if isinstance(obj, list):  return [cj(v) for v in obj]
    return obj

@app.route('/')
def index():
    return send_from_directory(os.path.join(BASE,'static'), 'index.html')

@app.route('/api/health')
def health():
    pkg = get_model(); a = get_analysis()
    return jsonify({'status':'ok','model':pkg['name'],'auc':a['meta']['best_auc']})

@app.route('/api/analysis')
def analysis():
    return Response(json.dumps(cj(get_analysis()),separators=(',',':')), mimetype='application/json')

@app.route('/api/predict', methods=['POST'])
def predict():
    body = request.get_json(force=True)
    if not body: return jsonify({'detail':'No JSON body'}), 400
    required = ['amount','hour','day_of_week','merchant_category','distance_from_home',
                'distance_from_last','ratio_to_median','pin_used','online_order',
                'repeat_retailer','account_age_days','transactions_24h','transactions_7d']
    for k in required:
        if body.get(k) is None: return jsonify({'detail':'Missing: '+k}), 422
    cat = str(body['merchant_category']).lower().strip()
    if cat not in MERCHANT_CATS:
        return jsonify({'detail':'merchant_category must be one of: '+str(MERCHANT_CATS)}), 400

    pkg=get_model(); model=pkg['model']; scaler=pkg['scaler']; le=pkg['le']
    cat_enc = int(le.transform([cat])[0])
    row = np.array([[
        float(body['amount']), int(body['hour']), int(body['day_of_week']), cat_enc,
        float(body['distance_from_home']), float(body['distance_from_last']),
        float(body['ratio_to_median']), int(body['pin_used']),
        int(body['online_order']), int(body['repeat_retailer']),
        float(body['account_age_days']), int(body['transactions_24h']),
        int(body['transactions_7d']),
        float(body.get('v1',0)), float(body.get('v2',0)), float(body.get('v3',0)),
        float(body.get('v4',0)), float(body.get('v5',0)),
    ]])
    prob = float(model.predict_proba(scaler.transform(row))[0,1])
    pred = 'FRAUD' if prob >= 0.5 else 'LEGITIMATE'
    risk = 'CRITICAL' if prob>=0.8 else 'HIGH' if prob>=0.5 else 'MEDIUM' if prob>=0.2 else 'LOW'

    a = get_analysis(); imp_vals = a['importance'][a['meta']['best_model']]['values']
    raw = row[0].tolist()
    contribs = sorted([{
        'feature':FEAT_NAMES[i], 'value':raw[i],
        'contribution':float((raw[i]-MEAN_VALS[i])/(abs(MEAN_VALS[i])+1e-8)*imp_vals[i])
    } for i in range(len(raw))], key=lambda x:abs(x['contribution']), reverse=True)

    return jsonify({'fraud_probability':round(prob,4),'prediction':pred,'risk_level':risk,
                    'model_used':pkg['name'],'top_features':contribs[:5]})

@app.route('/api/example/fraud')
def ex_fraud():
    return jsonify({'amount':1850.0,'hour':3,'day_of_week':6,'merchant_category':'online',
        'distance_from_home':420.0,'distance_from_last':850.0,'ratio_to_median':12.5,
        'pin_used':0,'online_order':1,'repeat_retailer':0,'account_age_days':22.0,
        'transactions_24h':14,'transactions_7d':38,'v1':-2.1,'v2':1.8,'v3':-1.5,'v4':1.2,'v5':-0.9})

@app.route('/api/example/legit')
def ex_legit():
    return jsonify({'amount':42.50,'hour':14,'day_of_week':2,'merchant_category':'grocery',
        'distance_from_home':2.0,'distance_from_last':1.5,'ratio_to_median':0.9,
        'pin_used':1,'online_order':0,'repeat_retailer':1,'account_age_days':1200.0,
        'transactions_24h':1,'transactions_7d':8,'v1':0.1,'v2':-0.2,'v3':0.3,'v4':-0.1,'v5':0.2})

@app.route('/static/<path:fn>')
def static_files(fn):
    return send_from_directory(os.path.join(BASE,'static'), fn)

if __name__ == '__main__':
    try:
        a=get_analysis(); pkg=get_model()
        print('='*50)
        print('  FraudGuard is running!')
        print(f"  Model : {pkg['name']}")
        print(f"  AUC   : {a['meta']['best_auc']:.4f}")
        print('  Open  : http://localhost:8000')
        print('='*50)
    except Exception as e:
        print(f'Warning: {e}')
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT',8000)), debug=False)
