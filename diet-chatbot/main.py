# diet-chatbot/main.py
# 간단한 다이어트 상담용 콘솔 챗봇

import random

EXIT_WORDS = ["exit", "quit", "종료", "그만", "bye", "안녕"]

RULES = [
    {
        "name": "인사",
        "keywords": ["안녕", "hello", "hi"],
        "responses": [
            "안녕하세요, 다이어트 챗봇입니다. 오늘은 어떤 식단이나 고민이 있으신가요?",
            "반가워요. 식단, 간식, 배고픔 관리 관련해서 편하게 질문해 주세요."
        ],
    },
    {
        "name": "배고픔_야식",
        "keywords": ["배고", "야식", "출출"],
        "responses": [
            "지금 배가 고프시군요. 가능한 한 물 먼저 한 잔 드시고, 배가 정말 고픈지 천천히 확인해 보세요.",
            "야식이 당길 땐, 과자나 빵 대신 삶은 계란, 두부, 채소 같은 가벼운 단백질 위주로 선택하는 것이 좋습니다.",
        ],
    },
    {
        "name": "간식_추천",
        "keywords": ["간식", "끼니 사이", "중간에"],
        "responses": [
            "간식은 하루 12번 정도로 정해두고, 단백질이 들어간 간식을 선택해 보세요. 예: 요거트, 삶은 계란, 견과류 소량 등.",
            "간식은 포만감을 주면서도 양을 과하게 늘리지 않는 것이 포인트입니다. 주로 단백질+식이섬유 조합을 추천합니다.",
        ],
    },
    {
        "name": "탄수화물_질문",
        "keywords": ["탄수", "빵", "밥", "면"],
        "responses": [
            "탄수화물은 완전히 끊기보다, 한 끼 기준으로 양을 정해두고 과하지 않게 조절하는 편이 유지에 더 좋습니다.",
            "빵, 면 같은 단순 탄수화물보다는 현미밥, 잡곡밥, 통곡물 빵처럼 천천히 소화되는 탄수화물을 우선으로 선택해 보세요.",
        ],
    },
    {
        "name": "단백질_질문",
        "keywords": ["단백질", "프로틴", "닭가슴살"],
        "responses": [
            "단백질은 포만감 유지와 근손실 방지에 중요합니다. 한 끼에 단백질 식품(닭가슴살, 생선, 계란, 두부 등)을 하나 이상 포함해 보세요.",
            "운동을 병행한다면 매 끼니에 단백질을 꾸준히 넣어 주는 것이 체지방 관리에 도움이 됩니다.",
        ],
    },
    {
        "name": "체중_정체기",
        "keywords": ["정체기", "체중이 안", "몸무게가 안"],
        "responses": [
            "체중 정체기는 흔하게 나타납니다. 최근 12주간 섭취량과 운동량이 크게 바뀌지 않았다면, 한 번만 더 12주 정도 유지해 보는 것도 방법입니다.",
            "수분, 숙변, 생리 주기 등으로 인해 단기 체중은 흔들립니다. 같은 조건(아침 공복 등)에서 1주 평균 체중을 비교해 보세요.",
        ],
    },
    {
        "name": "기본_다이어트_원칙",
        "keywords": ["다이어트", "감량", "체지방"],
        "responses": [
            "기본 원칙은 일정한 수면, 규칙적인 식사, 꾸준한 활동량입니다. 극단적으로 줄이기보다 '유지 가능한 정도'를 목표로 해 보세요.",
            "단기 목표보다, 내가 6개월 이상 유지할 수 있는 식단과 운동 패턴인지가 중요합니다. 너무 힘든 방법은 오래가기 어렵습니다.",
        ],
    },
]


def normalize(text: str) -> str:
    return text.strip().lower()


def find_rule(user_input: str):
    """단순 키워드 매칭 기반 룰 선택"""
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
            "말씀해 주신 내용이 조금 애매하게 들려요.\n"
            "• 식단 조절\n• 간식 선택\n• 체중 정체기\n이 중에서 어느 쪽에 가까운지 한 번 더 구체적으로 적어 주실 수 있을까요?"
        )

    return random.choice(rule["responses"])


def main():
    print("=== 다이어트 챗봇 ===")
    print("간단한 다이어트/식단 관련 상담을 도와드립니다.")
    print("대화를 종료하려면 '종료', '그만', 'exit' 등을 입력하세요.\n")

    while True:
        user_input = input("YOU : ").strip()

        if not user_input:
            continue

        if any(word in user_input.lower() for word in EXIT_WORDS):
            print("BOT : 오늘 대화는 여기까지 할게요. 수고하셨습니다.")
            break

        bot_answer = generate_response(user_input)
        print(f"BOT : {bot_answer}\n")


if __name__ == "__main__":
    main()