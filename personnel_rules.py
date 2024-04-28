from durable.lang import *

with ruleset('personnel_rules'):
    auto_id = 1
    prereq_id = 1
    auto_assign_id = 1
    subsequent_id = 1

    # 근무 점수 설정 규칙
    @when_all(m.service_period >= 0)
    def calculate_score(c):
        c.m['score'] = c.m['service_period'] * 10  # 근무 기간에 따라 근무 점수 계산
        print(f'{c.m["name"]}님의 근무 점수는 {c.m["score"]}입니다.')


    # 선수기 대상자 규칙
    @when_all((m.rank == '경') | (m.rank == '령') | (m.rank == '정'),
              m.desired_department == None)  # 희망 관서 입력 필요$$$
    def prerequisite_candidate(c):
        global prereq_id
        c.m['placement_order'] = prereq_id
        c.m['placement_type'] = '선수기 대상자'
        prereq_id += 1
        print(f'{c.m["name"]}님이 {c.m["placement_type"]}로 {c.m["placement_order"]}번째로 선발되었습니다. 근무 점수: {c.m["score"]}')

    # 자동배치 대상자 규칙
    @when_all(((m.rank == '사') | (m.rank == '교') | (m.rank == '장') | (m.rank == '위')) and
              m.service_period >= 18)  # 희망 관서 입력 필요
    def automatic_assignment_candidate(c):
        global auto_assign_id
        c.m['score'] = c.m['service_period'] * 10
        c.m['placement_order'] = auto_assign_id
        c.m['placement_type'] = '자동배치 대상자'
        auto_assign_id += 1
        print(f'{c.m["name"]}님이 {c.m["placement_type"]}로 {c.m["placement_order"]}번째로 선발되었습니다. 근무 점수: {c.m["score"]}')

    # 자동배치 점수 부족 대상자 규칙
    @when_all(((m.rank == '사') | (m.rank == '교') | (m.rank == '장') | (m.rank == '위')) and
              m.service_period < 18)  # 희망 관서 입력 필요
    def automatic_assignment_candidate(c):
            print(f'{c.m["name"]}님은 자동배치 대상자로 탈락되었습니다. (근무 개월 부족)')


    # 후수기 대상자 규칙
    @when_all(((m.rank == '사') | (m.rank == '교') | (m.rank == '장') | (m.rank == '위') )and
              m.personnel_type == '고충')  # 희망 관서 입력 필요
    def subsequent_candidate(c):
        global subsequent_id
        c.m['placement_order'] = subsequent_id
        c.m['placement_type'] = '후수기 대상자'
        subsequent_id += 1
        print(f'{c.m["name"]}님이 {c.m["personnel_type"]} 인사구분이 있으므로 {c.m["placement_type"]}로 {c.m["placement_order"]}번째로 선발되었습니다. ')

    # 근무 기간이 6년 이상인 경우 자동배치 대상자 선정 규칙
    @when_all((m.service_period > 72) and (m.desired_department == None))
    def long_service_period_candidate(c):
        global auto_assign_id
        c.m['placement_order'] = auto_assign_id
        c.m['placement_type'] = '자동배치(장기근무) 대상자'
        auto_assign_id += 1
        print(f'{c.m["name"]}님이 {c.m["placement_type"]}로 {c.m["placement_order"]}번째로 선발되었습니다. (희망관서 미선택)')

    # 희망관서 입력 규칙
    @when_all((m.desired_department == '경기') | (m.desired_department == '충남') |
              (m.desired_department == '충북') | (m.desired_department == '부산'))
    def department_input_rule(c):
        print(f'{c.m["name"]}님이 {c.m["desired_department"]}를 선택하였습니다.')

    # 희망관서 미입력 시 메시지 표시 규칙
    @when_all((m.desired_department == None ) and (m.service_period > 72))
    def missing_department_rule(c):
        print(f'{c.m["name"]}님은 희망 관서를 선택하지 않아서 자동배치 대상자로 선정되지 않았습니다.')


# 테스트 데이터
assert_fact('personnel_rules', {'rank': '사', 'score': 0, 'desired_department': '경기', 'service_period': 24, 'name': '김철수'})
assert_fact('personnel_rules', {'rank': '경', 'score': 0,  'desired_department': None, 'name': '이영희'})
assert_fact('personnel_rules', {'rank': '사', 'score': 0,  'desired_department': '충북', 'service_period': 3, 'name': '박민수'})
assert_fact('personnel_rules', {'rank': '사', 'score': 0,  'desired_department': '부산', 'personnel_type': '고충', 'name': '정수진'})
assert_fact('personnel_rules', {'rank': '사', 'score': 0,  'desired_department': None, 'service_period': 82, 'name': '지범수'})
