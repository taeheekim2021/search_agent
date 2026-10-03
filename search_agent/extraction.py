import re

from .domain import Conditions, Content

AGE_WORDS = {
    "한": 1,
    "두": 2,
    "세": 3,
    "네": 4,
    "다섯": 5,
    "여섯": 6,
    "일곱": 7,
    "여덟": 8,
    "아홉": 9,
    "열": 10,
}
TOPIC_ALIASES = {
    "공룡": ["티라노", "트리케라톱스"],
    "우주": ["행성", "별자리"],
    "수학": ["숫자", "덧셈"],
    "양치": ["이 닦", "치아"],
    "감정": ["기분"],
}


def extract(query: str, catalog: list[Content]) -> Conditions:
    ages = [int(age) for age in re.findall(r"(?<![\d.])(\d+)\s*(?:살|세)(?![가-힣])", query)]
    # Allow particles: '5살 아이', '5살에게', '만 5세가' are common queries.
    ages += [
        int(age) for age in re.findall(r"(?<![\d.])(\d+)\s*(?:살|세)(?=[가에게은는도])", query)
    ]
    for word, age in AGE_WORDS.items():
        if re.search(rf"(?<![가-힣]){word}\s*살", query):
            ages.append(age)
    if re.search(r"-\s*\d+\s*(?:살|세)|\d+\.\d+\s*(?:살|세)", query):
        raise ValueError("나이는 0~18 사이의 정수로 입력하세요.")
    if any(age > 18 for age in ages) or len(set(ages)) > 1:
        raise ValueError("한 번에 0~18세의 나이 하나만 지정하세요.")
    topics = sorted({topic for c in catalog for topic in c.topics})
    characters = sorted({character for c in catalog for character in c.characters})
    found_topics = [
        t for t in topics if t in query or any(alias in query for alias in TOPIC_ALIASES.get(t, []))
    ]
    found_characters = [c for c in characters if c in query]
    # Explicit unknown character requests should produce no matches, not unrelated results.
    explicit = re.findall(r"([가-힣A-Za-z0-9]+)\s*캐릭터", query)
    found_characters += [c for c in explicit if c not in found_characters]
    return Conditions(
        age=ages[0] if ages else None, topics=found_topics, characters=sorted(found_characters)
    )


def eligible(content: Content, conditions: Conditions) -> bool:
    return (
        (conditions.age is None or content.min_age <= conditions.age <= content.max_age)
        and all(topic in content.topics for topic in conditions.topics)
        and all(character in content.characters for character in conditions.characters)
    )
