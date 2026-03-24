"""Compact VQA demos for task00001."""

VQA_DEMOS = {
    "spatial": [
        {
            "qas": [
                {
                    "type": "spatial",
                    "question": "What objects are to the LEFT of brown basket?",
                    "answer": "kiwi, avocado",
                }
            ]
        },
        {
            "qas": [
                {
                    "type": "spatial",
                    "question": "Is red car inside any container?",
                    "answer": "no",
                }
            ]
        },
    ],
    "attribute": [
        {
            "qas": [
                {
                    "type": "attribute",
                    "question": "What is the color of red car?",
                    "answer": "red",
                }
            ]
        }
    ],
    "existence": [
        {
            "qas": [
                {
                    "type": "existence",
                    "question": "Is kiwi visible in the scene?",
                    "answer": "yes",
                }
            ]
        }
    ],
    "count": [
        {
            "qas": [
                {
                    "type": "count",
                    "question": "How many containers are visible?",
                    "answer": "2",
                }
            ]
        }
    ],
    "manipulation": [
        {
            "qas": [
                {
                    "type": "manipulation",
                    "question": "What action should happen next?",
                    "answer": "grasp",
                }
            ]
        }
    ],
}
