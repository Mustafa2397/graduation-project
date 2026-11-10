SEVERITY_SCORES = {
    "pothole": 90,
    "puddle": 60,
    "garbage": 40,
}

WEIGHT_SEVERITY = 0.40
WEIGHT_RECURRENCE = 0.35
WEIGHT_DURATION = 0.25
WEIGHT_LOCATION_CONTEXT = 0.0


def calculate_priority(
    problem_type: str,
    similar_reports_count: int,
    days_open: float,
    location_context_score: float = 0,
) -> float:
    if problem_type not in SEVERITY_SCORES:
        raise ValueError(
            f"نوع المشكلة '{problem_type}' غير معروف. الأنواع المسموحة: {list(SEVERITY_SCORES.keys())}"
        )

    severity_score = SEVERITY_SCORES[problem_type]
    recurrence_score = min(similar_reports_count / 10 * 100, 100)
    duration_score = min(days_open / 30 * 100, 100)

    priority = (
        severity_score * WEIGHT_SEVERITY
        + recurrence_score * WEIGHT_RECURRENCE
        + duration_score * WEIGHT_DURATION
        + location_context_score * WEIGHT_LOCATION_CONTEXT
    )
    return priority


if __name__ == "__main__":
    examples = [
        {
            "label": "مثال 1: حفرة قديمة وبلاغات كتير -> متوقع أولوية عالية جدًا",
            "inputs": {
                "problem_type": "pothole",
                "similar_reports_count": 8,
                "days_open": 25,
            },
        },
        {
            "label": "مثال 2: كيس قمامة، بلاغ واحد وعمره يوم -> متوقع أولوية منخفضة جدًا",
            "inputs": {
                "problem_type": "garbage",
                "similar_reports_count": 1,
                "days_open": 1,
            },
        },
        {
            "label": "مثال 3: بركة مياه، 3 بلاغات وعمرها 10 أيام -> متوقع أولوية متوسطة",
            "inputs": {
                "problem_type": "puddle",
                "similar_reports_count": 3,
                "days_open": 10,
            },
        },
        {
            "label": (
                "مثال 4: حفرة جديدة، بلاغ واحد ويوم واحد -> متوقع أولوية متوسطة "
                "لأن الخطورة عالية رغم قلة التكرار والمدة"
            ),
            "inputs": {
                "problem_type": "pothole",
                "similar_reports_count": 1,
                "days_open": 1,
            },
        },
    ]

    for example in examples:
        score = calculate_priority(**example["inputs"])
        print(example["label"])
        print(f"  المدخلات: {example['inputs']}")
        print(f"  الدرجة النهائية: {score:.2f}")
        print()
