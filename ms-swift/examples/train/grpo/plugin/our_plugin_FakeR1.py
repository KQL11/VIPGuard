import asyncio
import re
from typing import List

import json

from swift.plugin import ORM, orms
from swift.utils import get_logger

logger = get_logger()


# Code borrowed from plugin/orm.py
class MathAccuracy(ORM):

    def __init__(self):
        import importlib.util
        assert importlib.util.find_spec('math_verify') is not None, (
            "The math_verify package is required but not installed. Please install it using 'pip install math_verify'.")

    def __call__(self, completions, solution, **kwargs) -> List[float]:
        from latex2sympy2_extended import NormalizationConfig
        from math_verify import LatexExtractionConfig, parse, verify
        rewards = []
        for content, sol in zip(completions, solution):
            gold_parsed = parse(sol, extraction_mode='first_match', extraction_config=[LatexExtractionConfig()])
            if len(gold_parsed) != 0:
                # We require the answer to be provided in correct latex (no malformed operators)
                answer_parsed = parse(
                    content,
                    extraction_config=[
                        LatexExtractionConfig(
                            normalization_config=NormalizationConfig(
                                nits=False,
                                malformed_operators=False,
                                basic_latex=True,
                                equations=True,
                                boxed=True,
                                units=True,
                            ),
                            # Ensures that boxed is tried first
                            boxed_match_priority=0,
                            try_extract_without_anchor=False,
                        )
                    ],
                    extraction_mode='first_match',
                )
                # Reward 1 if the content is the same as the ground truth, 0 otherwise
                reward = float(verify(answer_parsed, gold_parsed))
            else:
                # If the gold solution is not parseable, we reward 1 to skip this example
                reward = 1.0
            rewards.append(reward)
        return rewards


class MathFormat(ORM):

    def __call__(self, completions, **kwargs) -> List[float]:
        """Reward function that checks if the completion has a specific format."""
        pattern = r'^<think>.*?</think>\s*<answer>.*?</answer>(?![\s\S])'
        matches = [re.match(pattern, content, re.DOTALL | re.MULTILINE) for content in completions]
        return [1.0 if match else 0.0 for match in matches]

class RealFakeFormat(ORM):

    def __call__(self, completions, **kwargs) -> List[float]:
        """Reward function that checks if the completion has a specific format."""
        pattern = r'^<think>.*?</think>\s*<answer>.*?</answer>(?![\s\S])'
        matches = [re.match(pattern, content, re.DOTALL | re.MULTILINE) for content in completions]
        return [0.2 if match else 0.0 for match in matches]

import re
from typing import List

# 预编译一次，提升速度
ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.IGNORECASE | re.DOTALL)
# 只保留 real / fake 单词
LABEL_RE  = re.compile(r"\b(real|fake)\b", re.IGNORECASE)


class RealFakeORM(ORM):
    def __call__(self, completions, solution, **kwargs) -> List[float]:
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


def check_snippet(pos_keywords, neg_keywords, think_content, 
                  max_reward=0.3, single_reward=0.1):
    """
    检查思考内容中的关键词，计算额外奖励
    
    Args:
        pos_keywords (list): 应该存在的关键词列表（正向奖励）
        neg_keywords (list): 不应该存在的关键词列表（负向惩罚）
        think_content (str): 思考过程的内容
        max_reward (float): 最大奖励值，默认为0.3
        single_reward (float): 每组完整的正向关键词的单次奖励，默认为0.1
    
    Returns:
        float: 额外的奖励分数
    """
    think_lower = think_content.lower()
    
    # 首先检查负向关键词（不应该存在的）
    # 如果发现任何一个负向关键词，直接返回最大惩罚
    for keyword in neg_keywords:
        if keyword.lower() in think_lower:
            return -max_reward  # 直接给予最大惩罚
    
    # 如果没有负向关键词，计算正向关键词的完整出现次数
    if not pos_keywords:  # 如果没有正向关键词，返回0
        return 0.0
    
    # 计算所有正向关键词都出现的最小次数
    min_complete_occurrences = float('inf')
    
    for keyword in pos_keywords:
        keyword_lower = keyword.lower()
        # 计算当前关键词在思考内容中出现的次数
        count = think_lower.count(keyword_lower)
        min_complete_occurrences = min(min_complete_occurrences, count)
    
    # 如果有任何关键词没有出现（count=0），则不给予任何奖励
    if min_complete_occurrences == float('inf') or min_complete_occurrences == 0:
        return 0.0  # 不完整出现时不记入奖励
    
    # 每次完整出现奖励 max_reward / 3（假设最多奖励3次完整出现）
    reward_per_complete = max_reward / 3
    total_reward = min_complete_occurrences * reward_per_complete
    
    # 限制最大奖励
    return min(max_reward, total_reward)

import re
class RealFakeORM_V2(ORM):
    def __call__(self, completions, solution, **kwargs):
        rewards = []
        # 既要有 think 标签，也要有 answer 标签，并且顺序正确
        # 修改正则表达式来同时捕获 think 和 answer 的内容
        full_pattern = re.compile(r'^<think>([\s\S]+?)</think>\s*<answer>(real|fake)</answer>$', re.IGNORECASE)
        for hyp, ref in zip(completions, solution):
            hyp_strip = hyp.strip()
            m = full_pattern.match(hyp_strip)
            if not m:
                # 缺少 think 或 answer，或格式不对
                rewards.append(-0.5)
                continue

            think_content = m.group(1).strip()  # 从 <think> 拿到思考过程内容
            pred_label = m.group(2).lower()     # 从 <answer> 拿到 real/fake
            
            # 这里可以对 think_content 进行进一步处理或分析
            # 例如：检查推理质量、关键词等

            gt_label = re.search(r'<answer>(real|fake)</answer>', ref, re.IGNORECASE).group(1).lower()
            
            # 基础奖励：预测正确性
            base_reward = 0.5 if pred_label == gt_label else -0.5
            
            # 分析思考内容，计算额外奖励
            if gt_label == 'real':
                pos_keywords = ['why real', 'if fake']
                neg_keywords = ['why fake', 'if real']
            else:  # gt_label == 'fake'
                pos_keywords = ['why fake', 'if real']
                neg_keywords = ['why real', 'if fake']
            
            # 使用check_snippet函数计算思考内容的额外奖励
            thinking_bonus = check_snippet(pos_keywords, neg_keywords, think_content, max_reward=0.3,
                                           single_reward=0.1)
            
            # 总奖励 = 基础奖励 + 思考奖励
            if thinking_bonus < 0:
                # 如果思考内容有负向关键词，直接给最大惩罚
                total_reward = -0.5
            else:
                # 否则，计算总奖励
                total_reward = base_reward + thinking_bonus
            rewards.append(total_reward)
                
        return rewards


class SmileORM(ORM):
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


class CountdownORM(ORM):

    def __call__(self, completions, target, nums, **kwargs) -> List[float]:
        """
        Evaluates completions based on Mathematical correctness of the answer

        Args:
            completions (list[str]): Generated outputs
            target (list[str]): Expected answers
            nums (list[str]): Available numbers

        Returns:
            list[float]: Reward scores
        """
        rewards = []
        for completion, gt, numbers in zip(completions, target, nums):
            try:
                # Check if the format is correct
                match = re.search(r'<answer>(.*?)<\/answer>', completion)
                if match is None:
                    rewards.append(0.0)
                    continue
                # Extract the "answer" part from the completion
                equation = match.group(1).strip()
                if '=' in equation:
                    equation = equation.split('=')[0]
                # Extract all numbers from the equation
                used_numbers = [int(n) for n in re.findall(r'\d+', equation)]

                # Check if all numbers are used exactly once
                if sorted(used_numbers) != sorted(numbers):
                    rewards.append(0.0)
                    continue
                # Define a regex pattern that only allows numbers, operators, parentheses, and whitespace
                allowed_pattern = r'^[\d+\-*/().\s]+$'
                if not re.match(allowed_pattern, equation):
                    rewards.append(0.0)
                    continue

                # Evaluate the equation with restricted globals and locals
                result = eval(equation, {"__builti'ns__": None}, {})
                # Check if the equation is correct and matches the ground truth
                if abs(float(result) - float(gt)) < 1e-5:
                    rewards.append(1.0)
                else:
                    rewards.append(0.0)
            except Exception:
                # If evaluation fails, reward is 0
                rewards.append(0.0)
        return rewards


class MultiModalAccuracyORM(ORM):

    def __call__(self, completions, solution, **kwargs) -> List[float]:
        """
        Reward function that checks if the completion is correct.
        Args:
            completions (list[str]): Generated outputs
            solution (list[str]): Ground Truths.

        Returns:
            list[float]: Reward scores
        """
        rewards = []
        from math_verify import parse, verify
        for content, sol in zip(completions, solution):
            reward = 0.0
            # Try symbolic verification first
            try:
                answer = parse(content)
                if float(verify(answer, parse(sol))) > 0:
                    reward = 1.0
            except Exception:
                pass  # Continue to next verification method if this fails

            # If symbolic verification failed, try string matching
            if reward == 0.0:
                try:
                    # Extract answer from solution if it has think/answer tags
                    sol_match = re.search(r'<answer>(.*?)</answer>', sol)
                    ground_truth = sol_match.group(1).strip() if sol_match else sol.strip()

                    # Extract answer from content if it has think/answer tags
                    content_match = re.search(r'<answer>(.*?)</answer>', content)
                    student_answer = content_match.group(1).strip() if content_match else content.strip()

                    # Compare the extracted answers
                    if student_answer == ground_truth:
                        reward = 1.0
                except Exception:
                    pass  # Keep reward as 0.0 if both methods fail
            rewards.append(reward)
        return rewards


# ref implementation: https://github.com/huggingface/open-r1/blob/main/src/open_r1/rewards.py
class CodeReward(ORM):

    def __init__(self):
        import importlib.util
        assert importlib.util.find_spec('e2b') is not None, (
            "The e2b package is required but not installed. Please install it using 'pip install e2b-code-interpreter'."
        )
        from dotenv import load_dotenv
        load_dotenv()

    @staticmethod
    def extract_code(completion: str, language: str) -> str:
        pattern = re.compile(rf'```{language}\n(.*?)```', re.DOTALL)
        matches = pattern.findall(completion)
        extracted_answer = matches[-1] if len(matches) >= 1 else ''
        return extracted_answer

    def run_async_from_sync(self, scripts: List[str], languages: List[str]) -> List[float]:
        """Function wrapping the `run_async` function."""
        # Create a new event loop and set it
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            # Run the async function and get the result
            rewards = loop.run_until_complete(self.run_async(scripts, languages))
        finally:
            loop.close()

        return rewards

    async def run_async(self, scripts: List[str], languages: List[str]) -> List[float]:
        from e2b_code_interpreter import AsyncSandbox

        # Create the sandbox by hand, currently there's no context manager for this version
        try:
            sbx = await AsyncSandbox.create(timeout=30, request_timeout=3)
        except Exception as e:
            logger.warning(f'Error from E2B executor: {e}')
            return [0.0] * len(scripts)
        # Create a list of tasks for running scripts concurrently
        tasks = [self.run_script(sbx, script, language) for script, language in zip(scripts, languages)]

        # Wait for all tasks to complete and gather their results as they finish
        results = await asyncio.gather(*tasks)
        rewards = list(results)  # collect results

        # Kill the sandbox after all the tasks are complete
        await sbx.kill()

        return rewards

    async def run_script(self, sbx, script: str, language: str) -> float:
        try:
            execution = await sbx.run_code(script, language=language, timeout=30)
        except Exception as e:
            logger.warning(f'Error from E2B executor: {e}')
            return 0.0
        try:
            return float(execution.text)
        except (TypeError, ValueError):
            return 0.0

    def __call__(self, completions, **kwargs) -> List[float]:
        """Reward function that evaluates code snippets using the E2B code interpreter.

        Assumes the dataset contains a `verification_info` column with test cases.
        """
        evaluation_script_template = """
        import subprocess
        import json

        def evaluate_code(code, test_cases):
            passed = 0
            total = len(test_cases)
            exec_timeout = 5

            for case in test_cases:
                process = subprocess.run(
                    ["python3", "-c", code],
                    input=case["input"],
                    text=True,
                    capture_output=True,
                    timeout=exec_timeout
                )

                if process.returncode != 0:  # Error in execution
                    continue

                output = process.stdout.strip()
                if output.strip() == case["output"].strip():
                    passed += 1

            success_rate = (passed / total)
            return success_rate

        code_snippet = {code}
        test_cases = json.loads({test_cases})

        evaluate_code(code_snippet, test_cases)
        """
        verification_info = kwargs['verification_info']
        languages = [info['language'] for info in verification_info]
        code_snippets = [
            self.extract_code(completion, language) for completion, language in zip(completions, languages)
        ]
        scripts = [
            evaluation_script_template.format(
                code=json.dumps(code), test_cases=json.dumps(json.dumps(info['test_cases'])))
            for code, info in zip(code_snippets, verification_info)
        ]
        try:
            rewards = self.run_async_from_sync(scripts, languages)

        except Exception as e:
            logger.warning(f'Error from E2B executor: {e}')
            rewards = [0.0] * len(completions)

        return rewards


class CodeFormat(ORM):

    def __call__(self, completions, **kwargs) -> List[float]:
        verification_info = kwargs['verification_info']
        rewards = []
        for content, info in zip(completions, verification_info):
            pattern = r'^<think>.*?</think>\s*<answer>.*?```{}.*?```.*?</answer>(?![\s\S])'.format(info['language'])
            match = re.match(pattern, content, re.DOTALL | re.MULTILINE)
            reward = 1.0 if match else 0.0
            rewards.append(reward)
        return rewards


orms['external_math_acc'] = MathAccuracy
orms['external_math_format'] = MathFormat
orms['external_countdown'] = CountdownORM
orms['external_r1v_acc'] = MultiModalAccuracyORM
orms['external_code_reward'] = CodeReward
orms['external_code_format'] = CodeFormat
orms['external_real_fake']= RealFakeORM
orms['external_format'] = RealFakeFormat
orms['external_real_fake_v2'] = RealFakeORM_V2
orms['external_smile'] = SmileORM

