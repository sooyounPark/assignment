# travel-chatbot/main.py
# 간단한 여행 상담용 콘솔 챗봇

import random

EXIT_WORDS = ["exit", "quit", "종료", "그만", "bye", "안녕"]

RULES = [
    {
        "name": "인사",
        "keywords": ["안녕", "hello", "hi", "여행"],
        "responses": [
            "안녕하세요, 여행 챗봇입니다. 어디로 떠날 계획이신가요?",
            "반가워요. 여행 준비, 짐 싸기, 일정 구성 같은 부분을 함께 정리해 볼까요?",
        ],
    },
    {
        "name": "짐싸기",
        "keywords": ["짐", "캐리어", "뭐 챙겨", "뭘 챙겨"],
        "responses": [
            "여행 짐은 '필수품(여권, 지갑, 카드, 약)', '의류', '전자기기', '세면용품' 순으로 체크리스트를 만들고 하나씩 확인해 보세요.",
            "날씨와 일정에 맞춰서 1일 1착 + 여벌 12벌 정도를 기준으로 옷을 챙기고, 나머지는 현지에서 조정하는 식이 부담이 덜합니다.",
        ],
    },
    {
        "name": "일정짜기",
        "keywords": ["일정", "코스", "동선", "스케줄"],
        "responses": [
            "하루에 큰 일정은 23개 정도로만 잡고, 나머지는 여유 시간을 남겨두는 것이 덜 지칩니다.",
            "이동 시간이 얼마나 걸리는지부터 확인한 뒤, 같은 방향에 있는 장소들을 묶어서 동선을 구성해 보세요.",
        ],
    },
    {
        "name": "비행기",
        "keywords": ["비행기", "항공", "공항"],
        "responses": [
            "국제선의 경우 공항에는 보통 출발 23시간 전에 도착하는 것이 안전합니다.",
            "장시간 비행이라면, 물을 자주 마시고, 간단한 스트레칭을 해 주면 도착 후 피로감이 덜합니다.",
        ],
    },
    {
        "name": "숙소",
        "keywords": ["숙소", "호텔", "에어비앤비", "airbnb"],
        "responses": [
            "숙소를 고를 때는, 후기에서 '위치', '소음', '청결' 부분을 우선적으로 확인해 보세요.",
            "야간 이동이 많다면, 역 근처나 주요 교통 허브 근처 숙소를 선택하면 체력 소모를 줄일 수 있습니다.",
        ],
    },
    {
        "name": "준비물_체크",
        "keywords": ["체크리스트", "준비물", "뭐 필요", "필수"],
        "responses": [
            "여행 기본 체크리스트 예시입니다:\n"
            "- 여권 / 신분증\n- 현지 결제 수단(카드, 현금)\n- 상비약(기본 진통제, 소화제 등)\n- 충전기 / 멀티탭\n- 일정이 적힌 메모 혹은 앱",
            "출국 전에는 여권 유효기간, 숙소 주소, 비상 연락처 정도를 메모에 정리해 두면 이동 중에 편합니다.",
        ],
    },
]


def normalize(text: str) -> str:
    return text.strip().lower()


def find_rule(user_input: str):
    normalized = normalize(user_input)
    matched_rules = []

    for rule in RULES:
        if any(keyword in normalized for keyword in rule["keywords"]):
            matched_rules.append(rule)

    if not matched_rules:
        return None
    return random.choice(matched_rules)


def generate_response(user_input: str) -> str:
    rule = find_rule(user_input)
    if rule is None:
        return (
            "어떤 종류의 여행 이야기인지 조금 더 구체적으로 적어 주세요.\n"
            "예: '짐 싸기', '일정 짜기', '공항 준비', '숙소 고르기' 등"
        )
    return random.choice(rule["responses"])


def main():
    print("=== 여행 챗봇 ===")
    print("여행 준비, 짐싸기, 일정 구성에 대해 간단히 상담해 드립니다.")
    print("대화를 종료하려면 '종료', '그만', 'exit' 등을 입력하세요.\n")

    while True:
        user_input = input("YOU : ").strip()

        if not user_input:
            continue

        if any(word in user_input.lower() for word in EXIT_WORDS):
            print("BOT : 즐거운 여행 준비 되시길 바랍니다. 안녕히 가세요.")
            break

        bot_answer = generate_response(user_input)
        print(f"BOT : {bot_answer}\n")


if __name__ == "__main__":
    main()