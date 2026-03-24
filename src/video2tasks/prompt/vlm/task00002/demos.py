"""Compact VQA demos for task00002."""

VQA_DEMOS = {
    "spatial": [
        {
            "qas": [
                {
                    "type": "spatial",
                    "question": "Is grapes inside black basket?",
                    "answer": "no",
                }
            ]
        },
        {
            "qas": [
                {
                    "type": "spatial",
                    "question": "Where is grapes relative to black basket?",
                    "answer": "left",
                }
            ]
        },
    ],
    "attribute": [
        {
            "qas": [
                {
                    "type": "attribute",
                    "question": "What is the color of black basket?",
                    "answer": "black",
                }
            ]
        }
    ],
    "existence": [
        {
            "qas": [
                {
                    "type": "existence",
                    "question": "Is grapes visible?",
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
                    "question": "How many baskets are visible?",
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
                    "question": "What is the target placement for grapes?",
                    "answer": "black basket",
                }
            ]
        }
    ],
}
