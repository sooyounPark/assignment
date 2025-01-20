
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sqlalchemy import create_engine
import urllib

# 데이터베이스 설정
DB_USER = "secuware"
DB_PASSWORD = urllib.parse.quote_plus("Secudb7700184!@#")
DB_HOST = "db.secuware.co.kr"
DB_PORT = "3306"
DB_NAME = "pas_test202250120"

DATABASE_URL = f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}?charset=utf8mb4"
engine = create_engine(DATABASE_URL, pool_pre_ping=True)

def load_data(query):
    """데이터베이스에서 데이터를 로드하는 함수"""
    try:
        data = pd.read_sql(query, engine)
        return data
    except Exception as e:
        print(f"[ERROR] 데이터 로드 중 오류 발생: {e}")
        return pd.DataFrame()

def preprocess_data(org_data, history_data, user_data):
    """사전 전처리를 수행"""
    # Label Encoding
    label_encoder = LabelEncoder()
    history_data['HST_TYP_ENC'] = label_encoder.fit_transform(history_data['HST_TYP'].astype(str))
    # Count Vectorizer
    vectorizer = CountVectorizer()
    vectorized_data = vectorizer.fit_transform(org_data['ORG_FL_NM'].astype(str))
    org_data['ORG_VECTOR'] = list(vectorized_data.toarray())

    return org_data, history_data, user_data, label_encoder, vectorizer

def initialize_resources():
    """필요한 데이터를 로드하고 사전 처리"""
    org_query = "SELECT ORG_ID, ORG_FL_NM, ORG_PRNT_ID FROM ptbl_org"
    history_query = """
    SELECT USER_ID, HST_ID, HST_ST_DT, HST_EDD_DT, HST_ORG_NM, HST_TYP
    FROM ptbl_history
    """
    user_query = "SELECT USER_ID, EMP_STAT, ORG_ID FROM ptbl_user"

    org_data = load_data(org_query)
    history_data = load_data(history_query)
    user_data = load_data(user_query)

    org_data, history_data, user_data, label_encoder, vectorizer = preprocess_data(org_data, history_data, user_data)

    return org_data, history_data, user_data, label_encoder, vectorizer