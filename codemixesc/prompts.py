"""Prompts.

Part 1 holds the prompts of MultiAgentESC exactly as in the released code (prompt.py and the
f-strings of multiagent.py, typos included); tests/test_agents.py checks them byte for byte
against external/MultiAgentESC when that checkout is present.

Part 2 holds the CodeMixESC additions. They are built by inserting a "Language register"
block (the profile R of the Code-Mix Profiler) and a fourth "language and cultural fit"
criterion into the base prompts, so the diff to the base paper stays small and visible.

Part 3 holds the baselines (few-shot CoT, translate-pivot).
"""
import json

from .profiler import describe_register, is_plain_english

# ============================================================== Part 1: MultiAgentESC (verbatim)
STRATEGY_DEFINITIONS = {
    "Question": "Asking for information related to the problem to help the user articulate the issues that they face. Open-ended questions are best, and closed questions can be used to get specific information.",
    "Restatement or Paraphrasing": "A simple, more concise rephrasing of the user's statements that could help them see their situation more clearly.",
    "Reflection of feelings": "Articulate and describe the user's feelings.",
    "Self-disclosure": "Divulge similar experiences that you have had or emotions that you share with the user to express your empathy.",
    "Affirmation and Reassurance": "Affirm the user's strengths, motivation, and capabilities and provide reassurance and encouragement.",
    "Providing Suggestions": "Provide suggestions about how to change, but be careful to not overstep and tell them what to do.",
    "Information": "Provide useful information to the user, for example with data, facts, opinions, resources, or by answering questions.",
    "Others": "Exchange pleasantries and use other support strategies that do not fall into the above categories.",
}
STRATEGIES = list(STRATEGY_DEFINITIONS)

SYSTEM = "You are a psychological counseling expert."
ADMIN_STRATEGY = "Initialize the group discussion about strategy selection."
ADMIN_DEBATE = "Initialize the group discussion about which response is the most appropriate."

BEHAVIOR_CONTROL = '### Instruction\nYou are a psychological counseling expert. You will be provided with an incomplete conversation between an Assistant and a User. \nPlease analyze whether this conversation reflects the user\'s current emotional state, the reason the user is seeking emotional support, and how the user plans to cope with the event. \nIf all three points are reflected, please reply "YES," otherwise reply "NO." \n\n### Conversation\n{context}\n\nYour answer must include two parts:\n1. "YES" or "NO"\n2. If "YES", briefly explain how the conversation reflects these elements; if "NO", explain which elements are missing.\n\nYour answer must follow this format:\n1. [YES or NO]\n2. [explaination]\n'

ZERO_SHOT = "### Instruction\nYou are a psychological counseling expert. You will be provided with a dialogue context between an 'Assistant' and a 'User'. Your task is to play a role as 'Assistant' and generate a response based on the given dialogue context. \n\n### Dialogue context\n{context}\n\nYour answer must be fewer than 30 words and must follow this format:\nResponse: [response]\n"

GET_EMOTION = "### Instruction\nYou are a psychological counseling expert. You will be provided with a dialogue context between an 'Assistant' and a 'User'. Please infer the emotional state expressed in the user's last utterance.\n\n### Dialogue context\n{context}\n\nYour answer must include the following elements:\nEmotion: the emotion user expressed in their last utterance.\nReasoning: the reasoning behind your answer.\n\nYour answer must follow this format: \nEmotion: [emotion]\nReasoning: [reasoning]\n"

GET_CAUSE = "### Instruction\nYou are a psychological counseling expert. You will be provided with a dialogue context between an 'Assistant' and a 'User'. Another agent analyzes the conversation and infers the emotional state expressed by the user in their last utterance. \n\n### Dialogue context\n{context}\n\n### Emotional state\n{emo_and_reason}\n\nPlease infer the specific event that led to the user's emotional state based on the dialogue context. Your answer must include the following elements:\nEvent: the specific event that led to the user's emotional state.\nReasoning: the reasoning behind your answer.\n\nYour answer must follow this format: \nEvent: [event]\nReasoning: [reasoning]\n"

GET_INTENTION = "### Instruction\nYou are a psychological counseling expert. You will be provided with a dialogue context between an 'Assistant' and a 'User'. Other agents have analyzed the conversation, infering the emotional state expressed by the user in their last utterance and the specific event that led to the user's emotional state. \n\n### Dialogue context\n{context}\n\n### Emotional state\n{emo_and_reason}\n\n### Event\n{cau_and_reason}\n\nPlease reasonably infer the user's intention based on the dialogue context, with the goal of addressing the event that lead to their emotional state. Your answer must include the following elements:\nIntention: user's intention which aims to address the event that lead to their emotional state.\nReasoning: the reasoning behind your answer.\n\nYour answer must follow this format: \nIntention: [intention]\nReasoning: [reasoning]\n"

SELECT_STRATEGY = "### You will be provided with a dialogue context between an 'Assistant' and a 'User'. Psychologists have analyzed the conversation, infering the emotional state expressed by the user in their last utterance, the specific event that led to the user's emotional state and user's intention aiming to address the event that lead to their emotional state.\n\n### Dialogue context\n{context}\n\n### Emotional state\n{emo_and_reason}\n\n### Event\n{cau_and_reason}\n\n### Intention\n{int_and_reason}\n\nBased on the provided information and dialogue context, please select a strategy for the 'Assistant' to generate an appropriate response, and explain why. Your strategy should differnet from others as much as possible.\nThe following are examples of different strategies, all presented in the format of <post\n[strategy] response>.\n\n### Examples\n{examples}\n\nYour answer must include the following elements:\nStrategy: Strategy for generating an response. The strategy must appear in the examples. Please choose different strategy from others as much as possible.\nReasoning: the reasoning behind your answer.\n\nYour answer must follow this format: \nStrategy: [strategy]\nReasoning: [reasoning]\n"

RESPONSE_WITH_STRATEGY = "You will be provided with a dialogue context between an 'Assistant' and a 'User'. Psychologists have analyzed the conversation, infering the emotional state expressed by the user in their last utterance, the specific event that led to the user's emotional state and user's intention aiming to address the event that lead to their emotional state.\n\n### Dialogue context\n{context}\n\n### Emotional state\n{emo_and_reason}\n\n### Event\n{cau_and_reason}\n\n### Intention\n{int_and_reason}\n\n\nPlease generate a response from the Assistant's perspective using the {strategy} strategy.\nThe following are examples of this strategy, all presented in the format of <post\n[strategy] response>.\n\n### Examples\n{examples}\n\nYour answer must be fewer than 30 words and must follow this format:\nResponse: [strategy] [response]\n"

DEBATE = "### You will be provided with a dialogue context between an 'Assistant' and a 'User'. Psychologists have analyzed the conversation, infering the emotional state expressed by the user in their last utterance, the specific event that led to the user's emotional state and user's intention aiming to address the event that lead to their emotional state.\n\n### Dialogue context\n{context}\n\n### Emotional state\n{emo_and_reason}\n\n### Event\n{cau_and_reason}\n\n### Intention\n{int_and_reason}\n\nBased on the provided information and dialogue context, please select the most appropriate response from the following options and explain why. \n\n### Response\n{responses_template}\n\nYour answer must include the following elements:\nResponse: the most appropriate response and the strategy used in this response.\nReasoning: the reasoning behind your answer.\n\nYour answer must follow this format: \nResponse: [strategy] [response]\nReasoning: [reasoning]"

DEBATE_PERSONA = 'You are a psychologist who is good at listening to others\' opinions and reflecting on your own thoughts. You are currently participating in a group discussion about which response is the most appropriate, and you are inclined to support the response "{response}". However, during the discussion, you need to carefully consider others\' perspectives and reflect on your own viewpoint, ultimately reaching a reliable answer.'

REFLECT = "### You will be provided with a dialogue context between an 'Assistant' and a 'User'. Psychologists have analyzed the conversation, infering the emotional state expressed by the user in their last utterance, the specific event that led to the user's emotional state and user's intention aiming to address the event that lead to their emotional state.\n\n### Dialogue context\n{context}\n\n### Emotional state\n{emo_and_reason}\n\n### Event\n{cau_and_reason}\n\n### Intention\n{int_and_reason}\n\nBased on the provided information and the context of the dialogue, a group discussion is taking place to determine which response is the most appropriate.\n\n### Discussion content\n{discussion_content}\n\nYou should carefully analyze the various different viewpoints above, reflect on your own thoughts, and ultimately arrive at a convincing result. Your thought can be changed if you believe the viewpoints of others are more reasonable.\n\nYour answer must include the following elements:\nResponse: the most appropriate response and the strategy used in this response.\nReasoning: the reasoning behind your answer.\n\nYour answer must follow this format: \nResponse: [strategy] [response]\nReasoning: [reasoning]"

REFLECT_PERSONA = 'You are a psychologist who is good at listening to others\' opinions and reflecting on your own thoughts. You are currently participating in a group discussion about which response is the most appropriate, and you are inclined to support the response "{response}". However, during the discussion, you need to carefully consider others\' perspectives and reflect on your own viewpoint, ultimately reaching a reliable answer. Your thought can be changed if you believe the viewpoints of others are more reasonable.'

JUDGE = "You will be provided with a dialogue context between an 'Assistant' and a 'User'. \n\n### Dialogue context\n{context}\n\nThe following are responses generated by the therapist using different strategies, all presented in the format of <[strategy] response>. Please select the most appropriate response and explain why.\n\n### Examples\n{template_responses}\n\nYour answer must include the following elements:\nResponse: the most appropriate response and the strategy used in this response.\nReasoning: the reasoning behind your answer.\n\nYour answer must follow this format: \nResponse: [strategy] [response]\nReasoning: [reasoning]"

SELF_REFLECTION = "You will be provided with a dialogue context between an 'Assistant' and a 'User'. \n\n### Dialogue context\n{context}\n\nThe following is a responses generated by the therapist using {pred_strategy} strategy, presented in the format of <[strategy] response>. Please analyze whether this response is consistent with the ongoing conversation, whether it aligns with the strategy, and whether it effectively helps alleviate the user's emotional stress.\n\n### Response\n[{pred_strategy}] {response}\n\nIf the respones meets the above requirements, please return it as is; if not, please modify the response and provide a more refined version. Refined version must less than 30 words.\n\nYour answer must include the following elements:\nResponse: original response or refined response and the strategy used in this response.\nReasoning: the reasoning behind your answer.\n\nYour answer must follow this format: \nResponse: [strategy] [origianl/refined response]\nReasoning: [reasoning]"

# ============================================================== Part 2: CodeMixESC additions
# Phrases taken from the proposal's draft prompts.
LANGUAGE_FIT = ("does the response match the user's language, script and level of Hindi-English mixing, "
                "and does it sound natural?")
HINGLISH_HINT = ("Many Indian help-seekers mix Hindi and English (Hinglish). Interpret common Hinglish distress "
                 "expressions (for example \"tension\", \"ghabrahat\", \"log kya kahenge\", \"dimaag kharab\", "
                 "\"mann nahi lagta\") in context, by what the user means, rather than translating them literally.")


def register_line(R):
    """One sentence stating the seeker's register profile R."""
    mix, script = describe_register(R)
    if is_plain_english(R):
        return f"The user writes {mix} in {script}."
    return f"The user writes {mix} in {script} (code-mixing index {R['cmi']:.2f})."


def reply_instruction(R):
    """What the reply must look like (proposal: 'Reply in the same script with a similar
    Hindi-English mix')."""
    if is_plain_english(R):
        return "Reply in plain English, as the user does."
    text = "Reply in the same script with a similar Hindi-English mix, the way the user would naturally type it in chat."
    if R.get("script") in ("Roman", None, "None"):
        text += " Write Hindi words in Roman script (for example 'bahut', 'pareshan'), never in Devanagari."
    return text


def register_instruction(R):
    """Profile plus reply instruction, for prompts that have no separate register block."""
    return register_line(R) + " " + reply_instruction(R)


def _register_block(R, hint=False):
    block = "### Language register\n" + register_line(R) + "\n"
    if hint and not is_plain_english(R):
        block += HINGLISH_HINT + "\n"
    return block + "\n"


def _insert_after(template, anchor, text):
    assert template.count(anchor) == 1, anchor
    return template.replace(anchor, anchor + text)


_CTX = "### Dialogue context\n{context}\n\n"
_INT = "### Intention\n{int_and_reason}\n\n"


def _esc(text):
    """Escape braces of text that is spliced into a .format() template."""
    return text.replace("{", "{{").replace("}", "}}")


def analysis_prompt(template, R):
    """Emotion / cause / intention prompts: the agents also see R and the Hinglish hint."""
    return _insert_after(template, _CTX, _esc(_register_block(R, hint=True)))


def zero_shot_register(R):
    return _insert_after(ZERO_SHOT, _CTX, _esc("### Language register\n" + register_instruction(R) + "\n\n"))


def response_with_strategy_register(R):
    t = _insert_after(RESPONSE_WITH_STRATEGY, _INT, _esc(_register_block(R)))
    t = t.replace("using the {strategy} strategy.\n",
                  "using the {strategy} strategy. " + _esc(reply_instruction(R)) + "\n")
    if not is_plain_english(R):
        t = t.replace("response>.\n\n### Examples",
                      "response>. The examples are written in English; use them only to understand the strategy, "
                      "not as a model of the language.\n\n### Examples")
    return t


def debate_register(R):
    t = _insert_after(DEBATE, _INT, _esc(_register_block(R)))
    return t.replace("from the following options and explain why. \n",
                     "from the following options and explain why. Besides how well a response fits the conversation, "
                     "the user's state and the support strategy, consider its language and cultural fit: "
                     + LANGUAGE_FIT + "\n")


def reflect_register(R):
    t = _insert_after(REFLECT, _INT, _esc(_register_block(R)))
    anchor = "if you believe the viewpoints of others are more reasonable.\n\nYour answer"
    assert t.count(anchor) == 1
    return t.replace(anchor, "if you believe the viewpoints of others are more reasonable. When judging the "
                             "responses, also consider their language and cultural fit: " + LANGUAGE_FIT +
                     "\n\nYour answer")


def self_reflection_register(R):
    t = _insert_after(SELF_REFLECTION, _CTX, _esc(_register_block(R)))
    t = t.replace("and whether it effectively helps alleviate the user's emotional stress.",
                  "whether it effectively helps alleviate the user's emotional stress, and whether it fits the "
                  "user's language and culture: " + LANGUAGE_FIT)
    return t.replace("Refined version must less than 30 words.",
                     "Refined version must less than 30 words and must match the user's language register.")


GATE = """You will be provided with a dialogue context between an 'Assistant' and a 'User'.

### Dialogue context
{context}

### Language register
{register_line}

The following response of the Assistant{strategy_clause} does not match the user's language: {problem}.

### Response
{tagged_response}

Rewrite the response so that it matches the user's language register. {reply_instruction} Keep its meaning{keep_strategy}, and make it sound natural. The rewritten response must be fewer than 30 words.

Your answer must follow this format:
Response: {format_tag}[rewritten response]"""


def gate_prompt(context, R, response, strategy, problem):
    has_strategy = strategy and strategy != "None"
    return GATE.format(
        context=context, register_line=register_line(R), problem=problem,
        strategy_clause=f" (which uses the {strategy} strategy)" if has_strategy else "",
        tagged_response=f"[{strategy}] {response}" if has_strategy else response,
        reply_instruction=reply_instruction(R),
        keep_strategy=f" and the {strategy} strategy" if has_strategy else "",
        format_tag="[strategy] " if has_strategy else "")


def describe_mismatch(R, reg):
    """Plain-language description of how a response register `reg` differs from R."""
    parts = []
    if reg["script"] not in ("None", R["script"]) and R["script"] != "None":
        parts.append(f"it is written in {reg['script']} script, while the user writes in {R['script']} script")
    hs = 0 if is_plain_english(R) else round(100 * R["hi_frac"])
    hr = round(100 * reg["hi_frac"])
    if hr < hs:
        parts.append(f"it has about {hr}% Hindi words, while the user's messages have about {hs}% (too little Hindi)")
    elif hr > hs:
        parts.append(f"it has about {hr}% Hindi words, while the user's messages have about {hs}% (too much Hindi)")
    return "; ".join(parts) or "its Hindi-English mix differs from the user's"


# ============================================================== Part 3: baselines
def strategy_definitions_text():
    text = "Here are 8 strategies for generating responses: \n\n"
    for k, v in STRATEGY_DEFINITIONS.items():
        text += f"{k}: {v}\n"
    return text


FEWSHOT_COT = """### Instruction
You are a psychological counseling expert. You will be provided with a dialogue context between an 'Assistant' and a 'User'. Your task is to play a role as 'Assistant' and generate a response based on the given dialogue context.

{strategy_definitions}
Think step by step before you answer: first infer the emotion expressed in the user's last utterance, then the specific event that led to it, then the user's intention; then choose the most suitable strategy from the list above and write the response with that strategy.

### Examples
{examples}

### Dialogue context
{context}

Your response must be fewer than 30 words, and your answer must follow this format:
Emotion: [emotion]
Event: [event]
Intention: [intention]
Strategy: [strategy]
Response: [response]
"""

FEWSHOT_EXAMPLE = """Dialogue context: {context}
Emotion: {emotion}
Event: {event}
Intention: {intention}
Strategy: {strategy}
Response: {response}"""

TRANSLATE_IN = """Translate the following conversation between a User and an Assistant from Roman-script Hinglish (Hindi-English code-mixing) into natural, fluent English. Translate every turn faithfully: keep the meaning, the emotions, names and the tone, and do not add, drop, merge or split turns. Turns that are already in English stay as they are.

Conversation:
{turns_json}

Return only JSON of the form {{"turns": [{{"id": 0, "text": "..."}}, ...]}} with exactly {n} turns and the same ids."""

TRANSLATE_OUT = """Translate the following reply of an emotional support Assistant from English into the user's language. {register_instruction} Keep the meaning, the support strategy and the warm tone, and keep it under 30 words.

### Reply
{response}

Your answer must follow this format:
Response: [translated reply]"""


def translate_in_prompt(turns, attempt=0):
    items = [{"id": i, "speaker": "User" if m["role"] == "user" else "Assistant", "text": m["content"]}
             for i, m in enumerate(turns)]
    p = TRANSLATE_IN.format(turns_json=json.dumps(items, ensure_ascii=False, indent=0), n=len(items))
    if attempt:
        p += f"\n\n(Attempt {attempt + 1}: return exactly {len(items)} turns with ids 0..{len(items) - 1}.)"
    return p
