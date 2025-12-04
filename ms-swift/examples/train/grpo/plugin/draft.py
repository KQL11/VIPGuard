import re
from typing import List

# 预编译一次，提升速度
ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.IGNORECASE | re.DOTALL)
# 只保留 real / fake 单词
LABEL_RE  = re.compile(r"\b(real|fake)\b", re.IGNORECASE)
# import Levenshtein


import re
class RealFakeORM_V2():
    def __call__(self, completions, solution, **kwargs):
        rewards = []
        # 既要有 think 标签，也要有 answer 标签，并且顺序正确
        full_pattern = re.compile(r'^<think>[\s\S]+?</think>\s*<answer>(real|fake)</answer>$', re.IGNORECASE)
        for hyp, ref in zip(completions, solution):
            hyp_strip = hyp.strip()
            m = full_pattern.match(hyp_strip)
            if not m:
                # 缺少 think 或 answer，或格式不对
                rewards.append(-0.5)
                continue

            pred_label = m.group(1).lower()   # 从 <answer> 拿到 real/fake
            gt_label = re.search(r'<answer>(real|fake)</answer>', ref, re.IGNORECASE).group(1).lower()
            if pred_label == gt_label:
                rewards.append(0.5)
            else:
                rewards.append(-0.5)
        return rewards


class SmileORM():
    def __call__(self, completions, solution, **kwargs):
        """Reward function that checks if the completion has a specific format."""
        pattern = r'^<think>.*?</think>\s*<answer>.*?</answer>(?![\s\S])'
        rewards = []
        for hyp, ref in zip(completions, solution):
            gt = ref.lower()
            gt = gt.split('<answer>')[-1].split('</answer>')[0].strip()
            pred = hyp.lower().strip()
            if not '<answer>' in pred or not '</answer>' in pred:
                reward = -0.5
            else:
                pred = pred.split('<answer>')[-1].split('</answer>')[0].strip()

                if gt == pred:
                    reward = 0.5  # 预测正确
                else:
                    reward = -0.5

                # else: reward 保持 0.0     # 预测错误

            rewards.append(reward)

        return rewards



comp = ["<think>\nOkay, let me analyze the image for potential manipulation or AI artifacts.\n*   **Observation 1:** The primary visual anomaly is the raccoon's left eye. It lacks a distinct pupil.\n*   **Analysis:**\n    *   While raccoons can have different eye shapes and expressions, the absence of a pupil is anatomically unusual and inconsistent with how natural eyes would appear.\n    *   This flaw disrupts the realism and biological plausibility of the eye.\n*   **Forensic Principle:** The absence of expected features, such as a pupil in an eye, is a common artifact in AI-generated images or a result of poor digital editing where an eye was likely added or manipulated.\n*   **Observation 2:** The raccoon's right leg/paw shows a distinct white patch.\n*   **Analysis:**\n    *   While color variations are present in natural fur, the white patch appears somewhat abrupt and lacks the integration expected with the surrounding fur and the animal's overall coloring.\n    *   It might look like an added patch or a poorly blended edit.\n*   **Forensic Principle:** Sudden color changes or patches that don't logically fit within a scene (like a single white spot on an animal's otherwise brown fur) can indicate digital alteration or an AI generation artifact where detail consistency is compromised.\n</think>\n<answer>n#CC[C@@H](O)C1cCcCC</answer>"]

sol = ['<answer>N#CC[C@@H](O)c1ccccc1</answer>']

x = SmileORM()
print(x(comp, sol))
