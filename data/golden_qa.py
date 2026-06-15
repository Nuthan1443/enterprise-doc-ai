"""
Golden QA set for RAGAS evaluation.
These are questions with known correct answers from hr_policy.pdf.
In production: 50+ questions reviewed by domain experts.
For portfolio: 3 questions is sufficient to demonstrate the framework.
"""

GOLDEN_QA = [
    {
        "question": "How many days of paid annual leave are employees entitled to?",
        "ground_truth": "All full-time employees are entitled to 18 days of paid annual leave per calendar year.",
    },
    {
        "question": "How many days in advance must leave be approved?",
        "ground_truth": "Leave must be approved by the direct manager at least 5 business days in advance.",
    },
    {
        "question": "How many days per week can employees work remotely?",
        "ground_truth": "Employees may work remotely up to 3 days per week subject to manager approval.",
    },
]