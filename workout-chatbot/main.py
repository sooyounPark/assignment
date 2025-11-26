# workout-chatbot/main.py
# 간단한 운동 상담용 콘솔 챗봇

import random

EXIT_WORDS = ["exit", "quit", "종료", "그만", "bye", "안녕"]

RULES = [
    {
        "name": "인사",
        "keywords": ["안녕", "hello", "hi"],
        "responses": [
            "안녕하세요, 운동 챗봇입니다. 현재 운동 루틴이나 고민이 있으신가요?",
            "반갑습니다. 운동 강도, 루틴 설계, 휴식 관련해서 무엇이든 물어보세요.",
        ],
    },
    {
        "name": "운동_시작",
        "keywords": ["운동 시작", "운동을 시작", "입문", "초보"],
        "responses": [
            "처음에는 너무 길게 하기보다, 주 23회, 2030분 정도의 가벼운 루틴부터 시작하는 것이 좋습니다.",
            "운동을 시작하실 때는 무게/속도보다 '부상 없이 꾸준히 할 수 있는지'를 먼저 확인해 보시는 걸 추천합니다.",
        ],
    },
    {
        "name": "근력운동",
        "keywords": ["근력", "웨이트", "무게", "스쿼트", "데드", "벤치"],
        "responses": [
            "근력운동은 자세 안정이 가장 우선입니다. 무게를 올리기보다는, 통증 없이 반복 가능한 무게에서 천천히 늘려 보세요.",
            "하루에 전신을 무리하게 모두 하기보다, 상체/하체/코어 등으로 나누어 회전시키는 방식도 도움이 됩니다.",
        ],
    },
    {
        "name": "유산소운동",
        "keywords": ["유산소", "러닝", "걷기", "조깅", "런닝"],
        "responses": [
            "유산소 운동은 숨이 차되, 대화는 가능한 정도의 강도부터 시작하는 것이 안전합니다.",
            "처음에는 시간보다 '빈도'를 목표로 잡으세요. 예를 들어, 일주일에 34번, 2030분 정도를 꾸준히 실천하는 식입니다.",
        ],
    },
    {
        "name": "근육통",
        "keywords": ["근육통", "알배김", "알이 배겼", "통증"],
        "responses": [
            "운동 후 근육통은 흔하지만, 관절이나 뼈 쪽 통증, 일상생활이 힘들 정도의 통증이면 강도를 낮추는 것이 좋습니다.",
            "같은 부위를 너무 연속으로 강하게 사용하기보다, 근육통이 심할 땐 다른 부위를 중심으로 운동하거나 가벼운 스트레칭 위주로 쉬어주세요.",
        ],
    },
    {
        "name": "휴식",
        "keywords": ["휴식", "쉬는 날", "rest", "회복"],
        "responses": [
            "휴식일은 근육과 컨디션 회복에 중요합니다. 완전 휴식일을 정하거나, 가벼운 스트레칭과 산책만 하는 날로 두는 것도 좋습니다.",
            "연속으로 고강도 운동을 하셨다면, 최소 12일 정도는 강도를 줄여서 몸 상태를 체크해 보세요.",
        ],
    },
    {
        "name": "운동_루틴",
        "keywords": ["루틴", "프로그램", "메뉴", "플랜"],
        "responses": [
            "루틴은 현재 체력과 주당 확보 가능한 시간을 먼저 정한 뒤, 그 안에서 상체/하체/코어/유산소를 어떻게 섞을지 설계하는 방식이 좋습니다.",
            "가장 중요한 것은 '너무 복잡하지 않고, 내가 반복할 수 있는 루틴'입니다. 우선 간단하게 만든 뒤, 점진적으로 수정해 보세요.",
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
            "어떤 종류의 운동에 대한 이야기인지 조금 더 알려주시면 좋을 것 같습니다.\n"
            "예: '근력운동 루틴', '유산소 강도', '운동 후 근육통' 같은 식으로 구체적으로 적어주세요."
        )
    return random.choice(rule["responses"])


def main():
    print("=== 운동 챗봇 ===")
    print("운동 루틴, 강도, 휴식에 대해 간단히 상담해 드립니다.")
    print("대화를 종료하려면 '종료', '그만', 'exit' 등을 입력하세요.\n")

    while True:
        user_input = input("YOU : ").strip()

        if not user_input:
            continue

        if any(word in user_input.lower() for word in EXIT_WORDS):
            print("BOT : 오늘 운동 이야기는 여기까지 할게요. 수고하셨습니다.")
            break

        bot_answer = generate_response(user_input)
        print(f"BOT : {bot_answer}\n")


if __name__ == "__main__":
    main()