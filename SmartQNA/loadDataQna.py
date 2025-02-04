import pandas as pd


def load_data_from_csv(file_path):
    """
    주어진 파일 경로의 CSV 파일을 로드하여 pandas DataFrame으로 반환하는 함수.

    반영사항:
    - CSV 파일을 불러오는 기능을 구현하여 데이터베이스 연결 없이 FAQ 데이터를 활용할 수 있도록 함.
    """
    try:
        data = pd.read_csv(file_path)
        print(f"[INFO] {file_path} 파일에서 데이터를 성공적으로 로드했습니다.")
        return data
    except Exception as e:
        print(f"[ERROR] CSV 파일 로드 중 오류 발생: {e}")
        return pd.DataFrame()