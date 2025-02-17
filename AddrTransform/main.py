from geopy.geocoders import Nominatim
import pandas as pd
import time
import re
from loadData import load_data  # load_data() 함수로 데이터 조회 (변경된 연결 방식 적용)

# AI 기반 주소 정규화 pipeline 초기화 (한 번만 로드)
from transformers import pipeline
normalizer = pipeline("text2text-generation", model="t5-small")

# 주소 전처리 함수: 괄호 안 내용 제거, 쉼표 제거, 공백 정리 등
def preprocess_address(address):
    # 괄호와 괄호 안의 내용을 제거
    cleaned = re.sub(r'\(.*?\)', '', address)
    # 쉼표를 공백으로 변경
    cleaned = cleaned.replace(',', ' ')
    # 불필요한 공백 제거
    cleaned = ' '.join(cleaned.split())
    return cleaned

# AI 기반 주소 정규화 함수
def ai_normalize_address(address):
    # 단순 기관명 등 약식 주소의 경우, 실제 위치 정보를 추론하도록 구체적으로 지시합니다.
    prompt = (
            "한국의 표준 주소 체계에 따라 아래 주소를 정규화해 주세요. "
            "입력된 주소가 기관명이나 약식 표현인 경우, 해당 기관의 실제 도로명 주소 및 지번 주소를 정확하게 찾아 반환해 주세요: " + address
    )
    result = normalizer(prompt, max_length=128, num_return_sequences=1)
    normalized_address = result[0]['generated_text']
    return normalized_address.strip()

# 주소를 좌표로 변환하는 함수 (무료 API 사용)
def convert_address(address):
    geolocator = Nominatim(user_agent="addr_transform")
    try:
        # 원본 주소로 시도
        location = geolocator.geocode(address, language="ko", timeout=10)
        if location:
            return location.latitude, location.longitude, location.address
        else:
            # 전처리한 주소로 재시도
            processed_address = preprocess_address(address)
            print(f"원본 주소 변환 실패. 전처리된 주소로 재시도: {processed_address}")
            location = geolocator.geocode(processed_address, language="ko", timeout=10)
            if location:
                return location.latitude, location.longitude, location.address
            else:
                # AI 기반 주소 정규화로 재시도
                normalized_address = ai_normalize_address(address)
                print(f"전처리된 주소도 실패. AI 정규화 주소로 재시도: {normalized_address}")
                location = geolocator.geocode(normalized_address, language="ko", timeout=10)
                if location:
                    return location.latitude, location.longitude, location.address
                else:
                    return None, None, None
    except Exception as e:
        print(f"주소 변환 에러 - {address}: {e}")
        return None, None, None

# 특정 user_id에 해당하는 주소 데이터를 조회하여 좌표 변환 후 CSV 파일로 저장하는 함수
def process_addresses_for_user():
    user_id = input("사용자 ID를 입력하세요: ").strip()
    # user_id에 해당하는 주소 데이터만 조회 (ADDR_NM이 NULL이 아닌 경우)
    select_query = f"SELECT user_id, ADDR_NM FROM ptbl_address WHERE user_id = '{user_id}' AND ADDR_NM IS NOT NULL"
    addresses = load_data(select_query)

    # DataFrame이 비어있는지 확인
    if addresses.empty:
        print("해당 사용자 ID에 해당하는 주소 데이터가 없습니다.")
        return

    results = []
    for row in addresses.itertuples(index=False):
        uid, addr = row
        lat, lng, formatted_addr = convert_address(addr)
        if lat and lng:
            print(f"user_id: {uid} / 주소: {addr}")
            print(f" -> 좌표: 위도={lat}, 경도={lng}")
            print(f" -> 변환된 주소: {formatted_addr}\n")
        else:
            print(f"user_id: {uid} / 주소: {addr} - 좌표 변환 실패\n")
        results.append({
            "user_id": uid,
            "original_addr": addr,
            "latitude": lat,
            "longitude": lng,
            "formatted_addr": formatted_addr
        })
        # 무료 API의 rate limit 고려: 1초 대기
        time.sleep(1)

    # 결과 리스트를 DataFrame으로 변환 후 CSV 파일로 저장
    df = pd.DataFrame(results)
    output_file = "output_addresses.csv"
    df.to_csv(output_file, index=False, encoding="utf-8-sig")
    print(f"\n변환된 결과가 CSV 파일({output_file})로 저장되었습니다.")

if __name__ == "__main__":
    process_addresses_for_user()