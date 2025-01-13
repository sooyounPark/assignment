import urllib
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

# 데이터베이스 설정
DB_USER = "secuware"
DB_PASSWORD = urllib.parse.quote_plus("Secudb7700184!@#")
DB_HOST = "db.secuware.co.kr"
DB_PORT = "3306"
DB_NAME = "pas_jn"

DATABASE_URL = f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}?charset=utf8mb4"
engine = create_engine(DATABASE_URL, pool_pre_ping=True)

# 데이터 로드
history_query = """
SELECT USER_ID, HST_ID, HST_ST_DT, HST_TYP, HST_ORG_NM
FROM ptbl_history
"""
org_query = """
SELECT ORG_ID, ORG_FL_NM, ORG_CD, ORG_PRNT_ID, ORG_PRNT_CD
FROM ptbl_org
"""

history_data = pd.read_sql(history_query, engine)
org_data = pd.read_sql(org_query, engine)

# 1. ptbl_history와 ptbl_org 간의 정확 매칭 확인
exact_match = pd.merge(
    history_data, org_data,
    left_on="HST_ORG_NM", right_on="ORG_FL_NM",
    how="left", indicator=True
)

# 2. 정확 매칭되지 않은 데이터 탐지
exact_anomalies = exact_match[exact_match["_merge"] == "left_only"]

# 3. 유사 매칭 수행 (LIKE 연산자 활용)
like_results = []
for _, row in exact_anomalies.iterrows():
    hst_org_nm = row["HST_ORG_NM"]
    like_query = text(f"""
    SELECT ORG_ID, ORG_FL_NM
    FROM ptbl_org
    WHERE ORG_FL_NM LIKE :hst_org_nm
    """)
    like_matches = pd.read_sql(like_query, engine, params={"hst_org_nm": f"%{hst_org_nm}%"})
    if not like_matches.empty:
        like_results.append({
            "USER_ID": row["USER_ID"],
            "HST_ID": row["HST_ID"],
            "HST_ORG_NM": row["HST_ORG_NM"],
            "MATCHED_ORG_FL_NM": like_matches.iloc[0]["ORG_FL_NM"],
            "ORG_ID": like_matches.iloc[0]["ORG_ID"]
        })
    else:
        like_results.append({
            "USER_ID": row["USER_ID"],
            "HST_ID": row["HST_ID"],
            "HST_ORG_NM": row["HST_ORG_NM"],
            "MATCHED_ORG_FL_NM": None,
            "ORG_ID": None
        })

like_anomalies = pd.DataFrame(like_results)

# 4. 이상 데이터 통합
final_anomalies = exact_anomalies[["USER_ID", "HST_ID", "HST_ORG_NM", "HST_ST_DT", "HST_TYP"]].copy()
final_anomalies = pd.merge(final_anomalies, like_anomalies, on=["USER_ID", "HST_ID", "HST_ORG_NM"], how="left")

# 5. 이상 데이터 탐지 (Isolation Forest)
numeric_features = ["HST_ST_DT"]
history_data["HST_ST_DT"] = pd.to_datetime(history_data["HST_ST_DT"], errors="coerce").map(lambda x: x.timestamp() if pd.notnull(x) else 0)

iso_forest = IsolationForest(contamination=0.05, random_state=42)
history_data["ANOMALY"] = iso_forest.fit_predict(history_data[numeric_features])

# 6. 이상 데이터 탐지 결과 결합
anomalies_with_iso = pd.merge(
    final_anomalies,
    history_data[["USER_ID", "HST_ID", "ANOMALY"]],
    on=["USER_ID", "HST_ID"],
    how="left"
)

# 7. 결과 저장
anomalies_with_iso.to_csv("org_history_anomalies.csv", index=False)
print("조직 및 이력 테이블에서 이상 데이터를 CSV로 저장하였습니다.")