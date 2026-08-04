from medical_svs_agent.reward import compute_score


def test_reward_requires_correct_tagged_answer():
    assert compute_score("medical-svs", "<answer>Benign.</answer>", "benign")["score"] == 1.0
    assert compute_score("medical-svs", "benign", "benign")["score"] == 0.0

